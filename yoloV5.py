!cd yolov5 && python train.py --img 320 --batch 32 --epochs 10 --data carScr_up.yaml --weights yolov5s.pt --cache --evolve



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images, targets, image_ids = next(iter(train_data_loader))
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, 
                            weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None
num_epochs = 2

loss_hist = Averager()
itr=1
for epoch in range(num_epochs):
    loss_hist.reset()
    for images, targets, image_ids, in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if itr % 50 == 0:
            print(f'Iteration #{itr} loss: {loss_value}')
        itr += 1
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
    print(f'Epoch #{epoch} loss: {loss_hist.value}')

