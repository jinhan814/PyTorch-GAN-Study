## Folder Structure

```
- model
-- model.py
-- network.py
-- custom_layer.py
- train.py
- trainer.py
- data_loader.py
```

## Things To Do

`2022-02-16`

- model.py에 GNet, DNet 합치는 부분 틀 잡아두기
- network에 Generator, Discriminator class 멤버 함수들 틀 잡아두기
- custom_layer에 EqualizedConv2d 등 network.py에서 필요한 함수 구현하기

`2022-02-17`

- model.py, GNet, DNet init 작성하기
- 다른 멤버함수는 프로토타입만 써두고 분배해서 따로 구현해오기
- 구현한 함수는 다음 스터디 때 코드리뷰하면서 합치기


`2022-03-09`
- minibatch stddev에 대해서 코드 완성 후 다시 보기