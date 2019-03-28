import torch
import torch.nn as nn
import torch.nn.functional as F



def similarity_loss(text_images, size_info, gamma1, gamma2, gamma3): # consider passing the perceptron layer
    """
    text_images pair: a batch of text_image pairs
    size_info: size of each dimension of image and text
    """
    loss1_w = 0
    loss2_w = 0
    loss_reg = 0

    M, H_r, H_w = size_info
    for text_image in text_images:
        e, v = text_image
        numerator, beta = calculate_matching_score(v, e, M, H_r, H_w, gamma1, gamma2)
        numerator = gamma3 * torch.exp(numerator)
        loss_reg += torch.norm(beta.mm(beta.t()) - torch.diag(beta.mm(beta.t())))
        P_DQ_denum = 0
        P_QD_denum = 0
        for e_sub, v_sub in text_images:
            denum, _ = calculate_matching_score(v, e_sub, M, H_r, H_w, gamma1, gamma2)
            denum2, _ = calculate_matching_score(v_sub, e, M, H_r, H_w, gamma1, gamma2)
            P_DQ_denum += gamma3 * torch.exp(denum)
            P_QD_denum += gamma3 * torch.exp(denum2)
        loss1_w -= numerator / P_DQ_denum
        loss2_w -= numerator / P_QD_denum
    return loss1_w + loss2_w + loss_reg


def calculate_matching_score(v, e, M, H_r, H_w, gamma1, gamma2):

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
    attn_score = F.softmax(gamma1 * normalized_similarity_matrix, dim=1)
    # each row of v_tidal represent the attention output
    v_tidal = attn_score.mm(v)
    R_QD = 0 # define matching score for one direction
    for i in range(e.size()[0]):
        R_QD += v_tidal[i].view(1, -1).mm(e[i].view(-1, 1)).squeeze() / (torch.norm(v_tidal[i], 2) * torch.norm(e[i], 2))
    R_QD = torch.log(torch.pow(R_QD, 1 / gamma2))


    # regard image box as query, might consider overflow
    similarity_matrix_copy = normalized_similarity_matrix.clone()
    similarity_matrix_copy = torch.exp(similarity_matrix_copy)

    # beta = torch.zeros(T, M)
    beta = []
    for i in range(M):
        local_sum = torch.sum(similarity_matrix_copy[:, i * H_r * H_w : (i + 1) * H_r * H_w])
        beta.append(torch.sum(torch.div(similarity_matrix_copy[:, i * H_r * H_w : (i + 1) * H_r * H_w], local_sum), dim=1))
    # beta = torch.tensor(beta)
    beta = torch.stack(beta, dim=0)
    # return torch.sum(beta)

    beta = beta.t()
    e_prime = (e.t()).mm(beta)

    # calculate R_QD in the second direction
    R_QD2 = 0
    for i in range(M):
        reference = 0
        for j in range(i * (H_r * H_w), (i + 1) * (H_r * H_w)):
            reference += v[j].view(1, -1).mm(e_prime[:, i].view(-1, 1)).squeeze() / (torch.norm(v[j], 2) * torch.norm(e_prime[:, i], 2))
        reference /= H_r * H_w
        R_QD2 += torch.exp(reference)
    R_QD2 = torch.log(torch.pow(R_QD2, 1 / gamma2))

    # add matching score for two directions.
    return R_QD + R_QD2, beta


# class version of similarity loss
class SimilarityLoss(nn.Module):

    def __init__(self, gamma1, gamma2, gamma3):
        super(SimilarityLoss, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

    def forward(self, text_images, size_info):
        return similarity_loss(text_images, size_info, gamma1, gamma2, gamma3)









if __name__ == "__main__":
    M = 2
    H_w = 10
    H_r = 10
    T = 5
    D = 20
    e = torch.rand(T, D, requires_grad=True)
    v = torch.rand(M , H_r , H_w, D, requires_grad=True)
    v1 = v.view(M * H_r * H_w, D)
    size_info = (M, H_r, H_w)
    gamma1 = 1
    gamma2 = 2
    gamma3 = 3
    loss = similarity_loss([(e, v1)], size_info, gamma1, gamma2, gamma3)
    loss.backward()
    print(e.grad)
    print(v1.grad)
    print(v.grad)

    print(loss)
    # x = torch.randn(3, 4, requires_grad=True)
    # y = torch.randn(3, 4, requires_grad=True)
    # loss = test_loss(x, y)
    # loss.backward()













