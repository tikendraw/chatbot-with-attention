# Chatbot with Attention

This repository contains the code for a chatbot with attention mechanism built using TensorFlow. The chatbot uses attention mechanism to focus on the most relevant parts of the input text during the response generation process.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.11 or higher

### Installation

1. Clone the repository:

```
git clone https://github.com/tikendraw/chatbot-with-attention.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```


### Usage

To train the chatbot on custom dataset, run the following command:
> dataset must have just two columns namely col1 and col2


```
python3 preprocess_dataset.py 'path_to_csv_dataset'
```
```
python3 training.py
```


To chat with the trained chatbot, run the following command:

```
python3 inference.py
```



## Acknowledgements

The code in this repository is based on the following resources:

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention-Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
