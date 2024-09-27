# ICNN

This repository is a pytorch implementation of Input convex neural network.

It is created by [Wang Ke], [Wu Ye-min], and [Yuan Hang]

## Notice
- **model**

The project now supports the model described in the jupyter notebook.

- **dataset**

The project now supports all xlsx in **Datasets**.


## Requirement

The environment should have all packages in [requirements.txt](./requirements.txt)

```bash
$ pip install -r requirements.txt
```

## Usage
Here, we take **jupyter notebook** as an example.

To get the training results, we can run:
```bash
$ model = ICNN(activ='relu', layers=[10, 500, 1], device='cuda:0')
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
i = 0
MSE_error = []
MAPE_error = []
while(True):
    y_pred = model(X)
    optimizer.zero_grad()
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    loss_MAPE = torch.mean(torch.abs(y_pred - y) / y)*100.0
    
    # 确保权重非负
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(min=0)
    i += 1
    if i % 1000 == 0:
        print("第{}次训练，MSE是{}, MAPE是{} ".format(i, loss.item(), loss_MAPE.item()))
        MSE_error.append(loss.item())
        MAPE_error.append(loss_MAPE.item())
    if loss.item() < 0.179:
        # 保存当前所有的网络参数
        torch.save(model.state_dict(), 'model.pth')
        print("训练满足精度要求，训练了{}次。".format(i))
        break
```

## Citation

If you use this project in your research, please cite it.

```
@inproceedings{EricICNN2024,
 title={Input Convex neural networks},
 author={Wang Ke and Wu Ye-min and Yuan Hang},
 booktitle={National Post-Graduate Mathematical Contest in Modeling},
 year={2024}
}
```



