# Train your language model course

We’ve all used Large Language Models (LLMs) and been amazed by what they can do. I wanted to understand how these models are built, so I created this course.

I’m from Morocco and speak Moroccan Darija. Most LLMs today understand it a little, but they can't hold proper conversations in Darija. So, as a challenge, I decided to train a language model from scratch using my own WhatsApp conversations in Darija.

I've made a YouTube playlist documenting every step. You can watch it at your own pace. If anything is unclear, feel free to open an issue in this repository. I’ll be happy to help!

[![course_thumbnail](./images/course_thumbnail%20.png)](https://www.youtube.com/playlist?list=PLMSb3cZXtIfptKdr56uEdiM5pR6HDMoUX)

## What is in this repository?

- `notebooks/`: Jupyter notebooks for each step in the pipeline.
- `Slides.odp`: Presentation slides used in the YouTube series.
- `data/`: Sample data and templates.
- `transformer/`: Scripts for the Transformer and LoRA implementations.
- `minbpe/`: A tokenizer from [Andrej' Karpathy's repo](https://github.com/karpathy/minbpe), since it's not available as a package.

## Setup

To get started, install [Python](https://www.python.org/downloads/) and the required dependencies by running:  

```bash
pip install -r requirements.txt
```

## What you will learn?

This course covers:  

1. Extracting data from WhatsApp.  
2. Tokenizing text using the BPE algorithm.  
3. Understanding Transformer models.  
4. Pre-training the model.  
5. Creating a fine-tuning dataset.  
6. Fine-tuning the model (Instruction tuning and LoRA fine-tuning).  

Each topic has a video in the [YouTube playlist](https://www.youtube.com/playlist?list=PLMSb3cZXtIfptKdr56uEdiM5pR6HDMoUX) and a Jupyter notebook in the [`notebooks/`](./notebooks/) folder.  

## My experience

Like you, I had never trained a language model before. After following the steps in this course, I built my own 42M parameter model called **BoDmagh**. In Moroccan Darija, **BoDmagh** can mean **someone with a brain**. The word **Bo + [noun]** means something is deep inside you, so **BoDmagh** can also mean a smart person.

Here are two example conversations I had with the model:  

![conversation_1](./images/conversation_1.png)
![conversation_2](./images/conversation_2.png)

The [Supervised Fine-Tuning (SFT) dataset](https://github.com/ImadSaddik/BoDmaghDataset) I created really helped improve the model’s ability to hold a conversation.  

### Limitations  

The model doesn’t always give correct answers. If I try to discuss many different topics, it struggles. This is likely because both the model and the SFT dataset are small. Training on more data and using a larger model could improve the results. I might explore this in the future.

## Contributions

We welcome contributions! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Need help?

You can reach me through:  

- **YouTube** – Leave a comment on the videos.  
- **LinkedIn** – [Connect with me](https://www.linkedin.com/in/imadsaddik/).  
- **Email** – [simad3647@gmail.com](mailto:simad3647@gmail.com).  
