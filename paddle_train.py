

def train():
    # feature_embedding T(W slice num) B(batch size) F_D(feature dem 512)
    pred_labels = preds.max(2)[1] # preds (slice num, batch nclass) raw_pred_label: slice num * batchsize
    pred_labels = pred_labels.squeeze(1).transpose(1,0).contiguous()
    b,t = pred_labels.shape
    feature_embedding = feature_embedding.transpose(1,0).contiguous()
    pred_labels = pred_labels.view(-1)
    feature_embedding = feature_embedding.view(b*t, -1)
    center_loss_ = center_loss(feature_embedding, pred_labels)
    H, cost = ctc_loss(preds, text, preds_size, length)
    total_cost = cost + center_loss_ * 0.25

