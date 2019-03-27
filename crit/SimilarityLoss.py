import torch
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
    for e, v in text_images:
        numerator, beta = torch.exp(calculate_matching_score(v, e, M, H_r, H_w, gamma1, gamma2))
        loss_reg += torch.norm(beta.mm(beta.t()) - torch.diag(beta.mm(beta.t())))
        P_DQ_denum = 0
        P_QD_denum = 0
        for e_sub, v_sub in text_images:
            P_DQ_denum += torch.exp(calculate_matching_score(v, e_sub, M, H_r, H_w, gamma1, gamma2))
            P_QD_denum += torch.exp(calculate_matching_score(v_sub, e, M, H_r, H_w, gamma1, gamma2))
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
    similarity_matrix = e.mm(v.to())
    # might consider overflow
    normalized_similarity_matrix = F.softmax(similarity_matrix, dim=0)

    #regard text as query, might consider overflow
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
    beta = torch.zeros(T, M)
    for i in range(M):
        local_sum = torch.sum(similarity_matrix_copy[:, i * H_r * H_w : (i + 1) * H_r * H_w])
        similarity_matrix_copy[:, i * H_r * H_w : (i + 1) * H_r * H_w] /= local_sum
        beta[:, i] = torch.sum(similarity_matrix_copy[:, i * H_r * H_w : (i + 1) * H_r * H_w], dim=1)
    e_prime = (e.t()).mm(beta)
    # calculate R_QD in the second direction
    R_QD2 = 0
    for i in range(M):
        reference = 0
        for j in range(i * (H_r * H_w), (i + 1) * (H_r * H_w)):
            reference += v[j].view(-1, 1).mm(e_prime[i].view(-1, 1)).squeeze() / (torch.norm(v[i], 2) * torch.norm(e_prime[i], 2))
        reference /= H_r * H_w
        R_QD2 += torch.exp(reference)
    R_QD2 = torch.log(torch.pow(R_QD2, 1 / gamma2))

    # add matching score for two directions.
    return R_QD + R_QD2, beta










if __name__ == "__main__":
    M = 2
    H_w = 10
    H_r = 10
    T = 5
    D = 20
    e = torch.randn(T, D)
    v = torch.randn(M, H_r, H_w, D)
    v = v.view(M * H_r * H_w, D)
    size_info = (M, H_r, H_w)
    gamma1 = 1
    gamma2 = 2
    gamma3 = 3
    loss = similarity_loss((e, v), size_info, gamma1, gamma2, gamma3)













