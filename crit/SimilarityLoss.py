import torch
import torch.nn.functional as F



def similarity_loss(text_images, gamma1, gamma2, gamma3): # consider passing the perceptron layer
    """
    text: D x T tensor
    imgs: a list of  D x L tensors (L = 289 in the paper)
    global_sent: D tensor
    global : D tensor
    """
    loss1_w = 0
    loss2_w = 0
    loss1_s = 0
    loss2_s = 0
    for text_image in text_images:
        R_QD = torch.zeros(1)
        R_DQ = torch.zeros(1)
        total_RQD = torch.zeros(1)
        total_RDQ = torch.zeros(1)
        total_RQD_global = torch.zeros(1)
        total_RDQ_global = torch.zeros(1)

        text, imgs, global_sentence, global_img = text_image
        R_QD_global = global_img.view(1, -1).mm(global_sentence.view(-1, 1))
        R_DQ_global = R_QD_global.clone()
        for img in imgs:
            relevance = calculate_relevance(img, text, gamma1, gamma2)
            R_QD = torch.exp(gamma3 * relevance)
            R_DQ += torch.exp(gamma3 * relevance)
            for text_image in text_images:
                text2, img2s, global_sentence2, global_img2 = text_image
                total_RQD += calculate_relevance(img, text2, gamma1, gamma2)
                total_RQD_global += global_img2.view(1, -1).mm(global_sentence.view(-1, 1))
                total_RDQ_global += global_img.view(1, -1).mm(global_sentence2.view(-1, 1))
                for img2 in img2s:
                    total_RDQ += calculate_relevance(img2, text, gamma1, gamma2)
            loss1_w -= torch.log(R_QD / total_RQD)
            total_RQD = torch.zeros(1)

        total_RDQ = total_RDQ / len(imgs)
        loss2_w -= torch.log(R_DQ / total_RDQ)
        loss1_s -= torch.log(R_QD_global / total_RQD_global)
        loss2_s -= torch.log(R_DQ_global / total_RDQ_global)

        return loss1_s + loss2_s + loss1_w + loss2_w


def calculate_relevance(img, text, gamma1, gamma2):
    D, T = text.size()
    _, L = img.size()
    similarity_matrix = text.t().mm(img)
    # normalize_base = torch.ones(similarity_matrix.size()[0], 1).mm(torch.sum(similarity_matrix, dim=1).view(1, -1))
    normalize_base = torch.sum(similarity_matrix, dim=1).view(-1, 1).mm(torch.ones(1, L))
    print("similarity_matrix size", similarity_matrix.size())
    print("normalize_base size", normalize_base.size())
    normalize_similarity_matrix = similarity_matrix / normalize_base
    normalize_similarity_matrix = gamma1 * normalize_similarity_matrix
    attn_distribution = F.softmax(normalize_similarity_matrix)

    region_context = torch.zeros(L, T)
    for i in range(T):
        
        region_context[:, i] = attn_distribution[i].mm(img)
    relevance = 0
    for t in range(text.size()[1]):
        relevance += torch.exp(region_context[:, t].view(1, -1).mm(text[:, t].view(-1, 1)) * gamma2)
    relevance = torch.log(torch.pow(relevance, 1 / gamma2))
    return relevance


if __name__ == "__main__":
    D = 10
    T = 5
    B = 3
    L = 20
    batch = 10
    sample = []
    for b in range(batch):
        text = torch.randn(D, T)
        imgs = []
        for i in range(B):
            imgs.append(torch.randn(D, L))
        global_sentent = torch.randn(D)
        global_img = torch.randn(D)
        sample.append([text, imgs, global_sentent, global_img])
    loss = similarity_loss(sample, 5, 10, 15)
    print(loss)
            


    






