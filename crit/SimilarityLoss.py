import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time

# class version of similarity loss
class SimilarityLoss(nn.Module):

    def __init__(self, gamma1, gamma2, gamma3, bsize=5):
        super(SimilarityLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.bsize = bsize


    def similarity_loss(self, image, text, length_info):  # consider passing the perceptron layer
        """
        text_images pair: a batch of text_image pairs
        size_info: size of each dimension of image and text
        """
        loss1_w = 0
        loss2_w = 0
        loss_reg = 0

        batch = image.size()[0]
        for i in range(batch):
            # print(i)
            e = text[i][:length_info[i]]  # remove zero padding
            v = image[i]
            M, D, H_r, H_w = v.size()
            v = v.permute(0, 3, 2, 1)
            v = v.contiguous().view(M * H_r * H_w, D)
            T, D = e.size()
            numerator, beta = self.calculate_matching_score(v, e, M, H_r, H_w)
            numerator = self.gamma3 * torch.exp(numerator)
            # tmp = beta.t().mm(beta)#beta.mm(beta.t())
            # loss_reg += torch.sqrt(torch.sum(tmp**2) - torch.sum(torch.diag(tmp)**2))
            P_DQ_denum = 0
            P_QD_denum = 0
            spinds = np.zeros(5,dtype=np.int32)
            spinds[0] = i
            spinds[1:] = np.random.choice(batch,4)
            # print('numerator'+str(numerator))
            for j in spinds:
                e_sub = text[j][:length_info[j]]
                v_sub = image[j].permute(0, 3, 2, 1)
                v_sub = v_sub.contiguous().view(M * H_r * H_w, D)
                denum, _ = self.calculate_matching_score(v, e_sub, M, H_r, H_w)
                denum2, _ = self.calculate_matching_score(v_sub, e, M, H_r, H_w)
                # print(denum.data)
                # print(denum2.data)
                P_DQ_denum += self.gamma3 * torch.exp(denum)
                P_QD_denum += self.gamma3 * torch.exp(denum2)
                # print('P_DQ_denum'+str(P_DQ_denum))
                # print('P_QD_denum'+str(P_QD_denum))
            
            loss1_w -= torch.log(numerator / P_DQ_denum)
            loss2_w -= torch.log(numerator / P_QD_denum)
            # print('loss1:'+str(loss1_w))
            # print('loss2:'+str(loss2_w))
        loss1_w = loss1_w/batch
        loss2_w = loss2_w/batch 
        # loss_reg = loss_reg/(batch*T*M)
        # print('loss_reg: '+str(loss_reg))
        return loss1_w + loss2_w #+ loss_reg

    def calculate_matching_score(self, v, e, lenAry, M, H_r, H_w):

        """
        calculate matching score of (Q, D) pair, consider bi-direction
        :param v: 3D tensor for img with dimension B x (M x H_r x W_r) x D
        :param e (text): 2D tensor for text with dimension Tb x D
        :param lengthArray with size B every value is the end index of every short sentence.
        :param gamma1: factor
        :param gamma2: factor
        :return: maching score for (Q, D) pair
        """

        def checkNan(var):
            assert(not bool((var != var).any()))
        # step 1 : concatnate: v:  B x (M x H_r x W_r) x D 
        B, _, D = v.size()
        Tb = e.size()[0]
        v = v.view(-1, D)
        # step 2: get s: (B x M x H_r x W_r) x Tb
        s = v.mm(e.t())
        # step 3: for loop calculating softmax in dimension Tb, output: (B x M x H_r x W_r) x Tb
        norm_temp = []
        prev = 0
        for idx in lenAry:
            norm_temp.append(F.softmax(s[:, prev: idx], dim=1))
            prev = idx
        s_nt = self.gamma1 * torch.cat(norm_temp, dim=1)  # (B x M x H_r x W_r) x Tb
        # step 4: softmax in dimension Vb reshape into 3 dimension and call softmax B x (M x H_r x W_r) x Tb, define (M x H_r x W_r) = Mhw

        alpha = F.softmax(s_nt.view(B, -1, Tb), dim=1) # B x Mhw x Tb
        # step 5: v (B x Mhw x D), \alpha (B x Mhw x Tb) --> v\tilde (B x Tb x D)
        v = v.view(B, -1, D)
        v_tidal = alpha.permute(0, 2, 1).bmm(v) # (B x Tb x D)
        # step 6: v\tilde (B x Tb x D), e (Tb x D) --> B x Tb
        v_tidal_norm = F.normalize(v_tidal, dim=2)
        e_norm = F.normalize(e, dim=1)
        logit_mat = torch.sum(v_tidal_norm * e_norm, dim=2) # B x Tb
        exp_mat = torch.exp(logit_mat)
        # step 7: for loop  B x Tb --> B x B
        score_temp = []
        prev = 0
        for idx in lenAry:
            score_temp.append(torch.sum(exp_mat[:, prev: idx], dim=1).unsqueeze(1))
            prev = idx
        score_mat = torch.cat(score_temp, dim=1)
        log_score_mat_1 = torch.log(torch.pow(score_mat, 1 / self.gamma2)+1e-10) # B x B 
        # print(log_score_mat_1.size())
        checkNan(log_score_mat_1)

        # step 1: reshape s  (B x M x H_r x W_r) x Tb -> B x M x (H x W) x Tb
        s_nt_2 = F.softmax(s.view(B,-1,Tb),dim=1)
        s_nt_2 = s_nt_2.view(B, M , -1, Tb) # B x M x (H x W) x Tb
        
        # step 2: generate s_d as denominator B x M x B
        s_exp = torch.exp(s_nt_2)
        sd = torch.sum(s_exp, dim=2)
        sd_tmp = []
        prev = 0
        for idx in lenAry:
            sd_tmp.append(torch.sum(sd[:,:,prev:idx],dim=2).unsqueeze(2).repeat(1,1,idx-prev))
            prev = idx
        sd = torch.cat(sd_tmp,dim=2) # B x M x Tb

        # step 3: compute beta B x M x Tb
        sd_rep = sd.unsqueeze(2).repeat(1,1,(H_r*H_w),1) # B x M x (H x W) x Tb
        beta = torch.sum(s_exp/sd_rep,dim=2) # B x M x Tb 

        # step 4: compute em_prime B x M x B x D
        beta = beta.view(-1, Tb)
        em_temp = []
        prev = 0
        for idx in lenAry:
            em_temp.append(beta[:, prev: idx].mm(e[prev: idx, :]))
            prev = idx
        em_prime = torch.stack(em_temp, dim=1).view(B, -1, B, D)  # B x M x B x D


        # step 5: compute score B x B

        em_prime_rep = em_prime.unsqueeze(2).repeat(1, 1, H_r * H_w, 1, 1)
        em_temp = F.normalize(em_prime_rep.view(-1, B, D),dim=2)
        v_temp = F.normalize(v.view(-1, D, 1),dim=1)
        logit_mat_2 = torch.sum(em_temp.bmm(v_temp).squeeze().view(B, M, H_r * H_w, B), dim=2) / (H_r * H_w) # B x M x B
        score_mat_2 = torch.pow(torch.sum(torch.exp(logit_mat_2), dim=1), 1 / self.gamma2)
        log_score_mat_2 = torch.log(score_mat_2+1e-10)
        checkNan(log_score_mat_2)

        # reg term
        beta = beta.view(B,M,Tb)
        beta_prime = beta.permute(0,2,1)
        tmp = beta.bmm(beta_prime) # B,M,M
        loss_reg = 0
        for i in range(B):
            loss_reg += torch.norm(tmp[i]-torch.diag(torch.diag(tmp[i])))
        loss_reg = loss_reg/B
        checkNan(loss_reg)
        # print('loss_reg: '+str(loss_reg.data))
        log_score_mat_1 = self.gamma3 * log_score_mat_1
        log_score_mat_2 = self.gamma3 * log_score_mat_2
        log_score_mat = log_score_mat_1 + log_score_mat_2

        match_qd = torch.diag(log_score_mat)
        pdq = -torch.sum(torch.log(match_qd/torch.sum(log_score_mat,dim=1)+1e-10))
        pqd = -torch.sum(torch.log(match_qd/torch.sum(log_score_mat,dim=0)+1e-10))
        loss1 = (pdq + pqd)/B
        checkNan(loss1)



        return loss1+loss_reg

        # B, T, _ = e.size()
        
        # expand_e = []
        # for i in range(e.size()[0]):
        #     expand_e.append(e[i].repeat(B, 1, 1))
        # expanded_e = torch.cat(expand_e, dim=0)
        # expanded_v = v.repeat(B, 1, 1)

        # similarity_matrix = expanded_e.bmm(expanded_v)  # generate B^2 x (M x H x W) x D matrix

        # # might consider overflow
        # normalized_similarity_matrix = F.softmax(similarity_matrix, dim=1)

        # # #regard text as query, might consider overflow
        # attn_score = F.softmax(self.gamma1 * normalized_similarity_matrix, dim=2)
        # # each row of v_tidal represent the attention output
        # v_tidal = attn_score.bmm(expanded_v)
        # R_QD = 0  # define matching score for one direction

        # # here the v_e_norm's dimension should be B^2 x T
        # v_e_norm = torch.mul(torch.norm(v_tidal, dim=2), torch.norm(expanded_e, dim=2))
        # # R_QD = torch.log(torch.pow(torch.sum(torch.exp(torch.diag(v_tidal.mm(e.t())) / v_e_norm * self.gamma2)), 1 / self.gamma2)+1e-10)
        # R_QD = torch.log(torch.pow(torch.sum(torch.exp(torch.sum(v_tidal * e, dim=1) / v_e_norm * self.gamma2)), 1 / self.gamma2)+1e-10)

        # # print('R_QD'+str(R_QD.data))
        # # regard image box as query, might consider overflow
        # similarity_matrix_copy = normalized_similarity_matrix.clone()
        # similarity_matrix_copy = torch.exp(similarity_matrix_copy)

        # # beta = torch.zeros(T, M)
        # time1 = time.time()
        # beta = []
        # for i in range(M):
        #     local_sum = torch.sum(
        #         similarity_matrix_copy[:, i * H_r * H_w: (i + 1) * H_r * H_w])
        #     beta.append(torch.sum(torch.div(
        #         similarity_matrix_copy[:, i * H_r * H_w: (i + 1) * H_r * H_w],
        #         local_sum), dim=1))
        # # print("1 ", time.time() - time1)
        # # beta = torch.tensor(beta)
        # beta = torch.stack(beta, dim=0)
        # # return torch.sum(beta)

        # beta = beta.t()
        # e_prime = (e.t()).mm(beta)

        # # calculate R_QD in the second direction
        # time2 = time.time()
        # R_QD2 = 0
        # for i in range(M):
        #     reference = 0
        #     v_local = v[i * (H_r * H_w) : (i + 1) * (H_r * H_w)]
        #     v_local_e = v_local.mm(e_prime[:, i].unsqueeze(1)).squeeze()
        #     # print('v_local_e'+str(v_local_e))
        #     v_local_e_norm = torch.sum(
        #         v_local_e / ((torch.norm(v_local, dim=1)+1e-10) * torch.norm(e_prime[:, i])))
        #     # print('v_local_e_norm'+str(v_local_e_norm.data))
        #     reference = v_local_e_norm / ( H_r * H_w)
        #     # print('reference'+str(reference))
        #     R_QD2 += torch.exp(reference)
        # # print("2 ", time.time() - time2)
        # R_QD2 = torch.log(torch.pow(R_QD2, 1 / self.gamma2)+1e-10)
        # # print('R_QD2'+str(R_QD2))
        # # add matching score for two directions.
        # return score_mat, log_score_mat

    
    # def generate_similarity_matrix(self, image, text, length_info):
    #     """
    #     Generate similarity matrix for evaluation
    #     """
    #     batch = image.size()[0]
    #     similarity_matrix = torch.zeros(batch, batch) # similarity matrix is a square matrix
    #     for i in range(batch):
    #         e = text[i][:length_info[i]]  # remove zero padding
    #         for j in range(batch):
    #             v = image[j]
    #             M, D, H_r, H_w = v.size()
    #             v = v.permute(0, 3, 2, 1)
    #             v = v.contiguous().view(M * H_r * H_w, D)
    #             score, _ = self.calculate_matching_score(v, e, M, H_r, H_w)
    #             similarity_matrix[i, j] = score
    #     return similarity_matrix




    def forward(self, image, text, length_info):
        """
        :param image: 3D tensor for img with dimension B x M x D x H_r x W_r
        :param text: 2D tensor for text with dimension B x 15 x D
        :param length_info with size B every value is the end index of every short sentence.
        """
        B, M, D, H_r, H_w = image.size()
        
        loss = 0
        # rand inds
        inds = list(range(B))
        random.shuffle(inds)
        image = image[inds].permute(0,1,3,4,2).contiguous().view(B,-1,D) #B x M x H_r x W_r x D
        text = text[inds]
        length_info = length_info[inds]

        numBlcks = int(B/self.bsize)+1
        for i in range(0,B,self.bsize):
            image_b = image[i:i+self.bsize]
            text_b = torch.cat([text[j][:length_info[j]] for j in range(i,min(i+self.bsize,B))],dim=0).contiguous()
            len_b = [torch.sum(length_info[i:j+1]) for j in range(i,min(i+self.bsize,B))]
            loss += self.calculate_matching_score(image_b, text_b, len_b, M, H_r, H_w)
        loss = loss/numBlcks
        return loss


if __name__ == "__main__":
    # M = 2
    # H_w = 10
    # H_r = 10
    # T = 5
    # D = 20
    # torch.manual_seed(7)
    # torch.backends.cudnn.deterministic=True
    # image = torch.randn(10, 1, 1024, 7, 7)

    # image.requires_grad = True
    # torch.manual_seed(7)
    # text = torch.randn(10, 15, 1024)
    # text.requires_grad = True
    # length_info = torch.tensor([10, 12, 11, 9, 8, 14, 5, 10, 7, 10])
    # m = SimilarityLoss(1, 1, 1)
    # loss = m(image, text, length_info)
    # matrix = m.generate_similarity_matrix(image, text, length_info)
    # print(matrix)
    # print(loss)
    # loss.backward()


    # torch.manual_seed(7)
    # torch.backends.cudnn.deterministic=True
    # image = torch.randn(5, 3*49, 1024)
    
    # image.requires_grad = True
    # torch.manual_seed(7)
    # text = torch.randn(30, 1024)
    # text.requires_grad = True
    # length_info = torch.tensor([5, 12, 18, 23, 30])
    # m = SimilarityLoss(0.5, 0.5, 1)
    # loss = m.calculate_matching_score(image, text, length_info, 3, 7, 7)
    # print('final loss: '+str(loss.data))
    # # matrix = m.generate_similarity_matrix(image, text, length_info)
    # # print(matrix)
    # # print(loss)
    # loss.backward()
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic=True
    image = torch.randn(78, 1, 1024,7,7)
    
    image.requires_grad = True
    torch.manual_seed(7)
    text = torch.randn(78, 15, 1024)
    text.requires_grad = True
    length_info = torch.tensor([5, 6, 8, 4, 7]*16)
    crit = SimilarityLoss(0.5, 0.5, 1)
    # loss = m.calculate_matching_score(image, text, length_info, 3, 7, 7)
    loss = crit(image,text,length_info)
    print('final loss: '+str(loss.data))
    # matrix = m.generate_similarity_matrix(image, text, length_info)
    # print(matrix)
    # print(loss)
    loss.backward()




