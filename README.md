# Lightweight MNIST Classifier

![Build Status](https://github.com/shettysaish20/MNIST-training-script/workflows/MNIST%20Model%20Tests/badge.svg)

This project implements a lightweight CNN model for MNIST digit classification that achieves high accuracy with minimal parameters.

## Model Specifications

- **Parameter Count**: Less than 25,000 parameters
- **Training Performance**: Achieves >95% accuracy in just 1 epoch
- **Architecture**: 
  - 2 Convolutional layers
  - Max pooling for dimension reduction
  - Single fully connected layer
  - Total parameters: ~9,098

## Requirements

- Python 3.8+
- PyTorch
- torchvision

Install dependencies: 
```
pip install torch torchvision
```

directory structure:
```
 your-repo/
   ├── .github/
   │   └── workflows/
   │       └── model_tests.yml
   ├── mnist_model.py
   └── test_mnist_model.py
```
To run tests:

```
python test_mnist_model.py
```


## GitHub Actions

The repository includes automated testing that verifies:
1. The model has less than 25,000 parameters
2. The model achieves >95% accuracy in one epoch

Tests run automatically on:
- Every push to main branch
- Every pull request

## Model Architecture Details

- Input Layer: 1x28x28 (MNIST image)
- Conv1: 8 filters (3x3)
- MaxPool: 2x2
- Conv2: 16 filters (3x3)
- MaxPool: 2x2
- Fully Connected: 784 → 10

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request