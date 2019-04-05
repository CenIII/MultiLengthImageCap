import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time

# class version of similarity loss
class SimilarityLoss(nn.Module):

    def __init__(self, gamma1, gamma2, gamma3):
        super(SimilarityLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3


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
            tmp = beta.mm(beta.t())
            loss_reg += torch.sqrt(torch.sum(tmp**2) - torch.sum(torch.diag(tmp)**2))
            P_DQ_denum = 0
            P_QD_denum = 0
            spinds = np.zeros(5,dtype=np.int32)
            spinds[0] = i
            spinds[1:] = np.random.choice(batch,4)
            for i in spinds:
                e_sub = text[i][:length_info[i]]
                v_sub = image[i].permute(0, 3, 2, 1)
                v_sub = v_sub.contiguous().view(M * H_r * H_w, D)
                denum, _ = self.calculate_matching_score(v, e_sub, M, H_r, H_w)
                denum2, _ = self.calculate_matching_score(v_sub, e, M, H_r, H_w)
                P_DQ_denum += self.gamma3 * torch.exp(denum)
                P_QD_denum += self.gamma3 * torch.exp(denum2)
            loss1_w -= numerator / P_DQ_denum
            loss2_w -= numerator / P_QD_denum

        loss1_w = loss1_w/batch
        loss2_w = loss2_w/batch 
        loss_reg = loss_reg/(batch*T*M)
        print('numerator'+str(numerator))
        print('P_QD_denum'+str(P_QD_denum))
        print('loss1:'+str(loss1_w))
        print('loss2:'+str(loss2_w))
        print('loss_reg: '+str(loss_reg))
        return loss1_w + loss2_w + loss_reg

    def calculate_matching_score(self, v, e, M, H_r, H_w):

        """
        calculate matching score of (Q, D) pair, consider bi-direction
        :param v: 2D tensor for img with dimension (M x H_r x W_r) x D
        :param text: 2D tensor for text with dimension (T x D)
        :param gamma1: factor
        :param gamma2: factor
        :return: maching score for (Q, D) pair
        """
        T, _ = e.size()
        similarity_matrix = e.mm(v.t())
        # return torch.sum(similarity_matrix)

        # might consider overflow
        normalized_similarity_matrix = F.softmax(similarity_matrix, dim=0)

        # #regard text as query, might consider overflow
        attn_score = F.softmax(self.gamma1 * normalized_similarity_matrix, dim=1)
        # each row of v_tidal represent the attention output
        v_tidal = attn_score.mm(v)
        R_QD = 0  # define matching score for one direction
        # for i in range(e.size()[0]):
        #     R_QD += torch.exp((v_tidal[i].view(1, -1).mm(e[i].view(-1, 1)).squeeze() / (
        #                 torch.norm(v_tidal[i], 2) * torch.norm(e[i], 2))) * self.gamma2)
        # R_QD = torch.log(torch.pow(R_QD, 1 / self.gamma2))


        R_QD = torch.log(torch.pow(torch.sum(torch.exp(torch.diag(v_tidal.mm(e.t())) * self.gamma2)), 1 / self.gamma2))

        # regard image box as query, might consider overflow
        similarity_matrix_copy = normalized_similarity_matrix.clone()
        similarity_matrix_copy = torch.exp(similarity_matrix_copy)

        # beta = torch.zeros(T, M)
        time1 = time.time()
        beta = []
        for i in range(M):
            local_sum = torch.sum(
                similarity_matrix_copy[:, i * H_r * H_w: (i + 1) * H_r * H_w])
            beta.append(torch.sum(torch.div(
                similarity_matrix_copy[:, i * H_r * H_w: (i + 1) * H_r * H_w],
                local_sum), dim=1))
        # print("1 ", time.time() - time1)
        # beta = torch.tensor(beta)
        beta = torch.stack(beta, dim=0)
        # return torch.sum(beta)

        beta = beta.t()
        e_prime = (e.t()).mm(beta)

        # calculate R_QD in the second direction
        time2 = time.time()
        R_QD2 = 0
        for i in range(M):
            reference = 0
            # def tmp():
            #     reference = 0
            #     for j in range(i * (H_r * H_w), (i + 1) * (H_r * H_w)):
            #         reference += self.gamma2 * v[j].view(1, -1).mm(
            #             e_prime[:, i].view(-1, 1)).squeeze() / (
            #                                 torch.norm(v[j]) * torch.norm(
            #                             e_prime[:, i]))
            #     reference = v_local_e_norm / ( H_r * H_w)
            #     return reference
            v_local = v[i * (H_r * H_w) : (i + 1) * (H_r * H_w)]
            v_local_e = v_local.mm(e_prime[:, i].unsqueeze(1)).squeeze()
            v_local_e_norm = torch.sum(
                v_local_e / (torch.norm(v_local, dim=1) * torch.norm(e_prime[:, i])))
            reference = v_local_e_norm / ( H_r * H_w)
            R_QD2 += torch.exp(reference)
        # print("2 ", time.time() - time2)
        R_QD2 = torch.log(torch.pow(R_QD2, 1 / self.gamma2))

        # add matching score for two directions.
        return R_QD + R_QD2, beta


    def forward(self, image, text, length_info):
        """
        image: batch x boxes x D x H x W
        text: batch x T x D
        length_info: batch x 1
        """
        return self.similarity_loss(image, text, length_info)


if __name__ == "__main__":
    # M = 2
    # H_w = 10
    # H_r = 10
    # T = 5
    # D = 20
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic=True
    image = torch.randn(10, 1, 1024, 7, 7)
    image.requires_grad = True
    torch.manual_seed(7)
    text = torch.randn(10, 15, 1024)
    text.requires_grad = True
    length_info = torch.tensor([10, 12, 11, 9, 8, 14, 5, 10, 7, 10])
    m = SimilarityLoss(1, 1, 1)
    loss = m(image, text, length_info)
    print(loss)
    loss.backward()
    # M = 2
    # H_w = 2
    # H_r = 2
    # T = 2
    # D = 2
    # e = torch.FloatTensor([[0.3, 0.8], [0.7, 0.2]])
    # e2 = e - 0.1
    # e.requires_grad = True
    # e2.requires_grad = True
    # v = torch.FloatTensor(
    #     [[0.5, 0.3], [0.5, 1.0], [0.6, 0.4], [0.2, 0.4], [0.7, 1.2], [0.6, 1.6],
    #      [1.1, 0.7], [0.5, 0.2]])
    # v2 = v + 0.2
    # v2.requires_grad = True
    # v.requires_grad = True
    # # e = torch.rand(T, D, requires_grad=True)
    # # v = torch.rand(M , H_r , H_w, D, requires_grad=True)
    # v1 = v.view(M * H_r * H_w, D)
    # size_info = (M, H_r, H_w)
    # gamma1 = 1
    # gamma2 = 2
    # gamma3 = 3
    # loss = similarity_loss([(e, v), (e2, v2)], size_info, gamma1, gamma2, gamma3)
    # loss.backward()
    # print(e.grad)
    # print(v1.grad)
    # print(v.grad)
    #
    # print(loss)
    # # x = torch.randn(3, 4, requires_grad=True)
    # # y = torch.randn(3, 4, requires_grad=True)
    # # loss = test_loss(x, y)
    # # loss.backward()













