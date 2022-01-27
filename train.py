def train():
    preds,feature_embedding = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #cost = criterion(preds, text, preds_size, length) / batch_size
    
    # === center loss ===
    # feature_embedding T(W slice num) B(batch size) F_D(feature dem 512)
    raw_pred_label = preds.max(2)[1] # preds (slice num, batch nclass) raw_pred_label: slice num * batchsize
    batch_embedding, batch_center_labels = raw_pred_to_features(raw_pred_label, length, labels, feature_embedding) 
    center_loss_ = None
    
    if batch_embedding is not None:
        # 计算center loss
        center_loss_ = center_loss(batch_embedding, batch_center_labels) # args1 batch*512 args2 batch * T(slice num)
    else:
        center_loss_ = 0
    
    total_cost = cost + center_loss_ * 0.00001


