from network import GNet

if __name__ == '__main__':
    generator = GNet()
    generator.AddScale(1024)
    print(generator)