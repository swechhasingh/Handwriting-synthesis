# Handwriting generation!

Hey! Welcome to Lyrebird! We're excited to have you here. Since we work a lot with neural networks, this task will introduce you to the kind of models that we work with everyday. If you get stuck, please contact me and I'll be happy to provide hints.

We will solve a fun problem: handwriting generation. This problem involves 2 sequences: a sequence of text (input) and a sequence of points related to the position of a pen (output). If you haven't done it before, please check out [this great paper](https://arxiv.org/pdf/1308.0850.pdf) by Alex Graves. When you finish this task, you should have your own personal scribe :)

```Note: There are many solutions for this task available online. Please, don't look at them while solving it. The purpose of this task is to get you started into the kind of work we do at Lyrebird. We care a lot about your thought processes, the ideas you have to solve difficult problems and your coding style. Trying to copy a solution available online defeats the purpose of this task. Furthermore, it will be very easy to realize if you didn't solve it on your own.```


### Data description:

There are 2 data files that you need to consider: `data.npy` and `sentences.txt`. `data.npy`contains 6000 sequences of points that correspond to handwritten sentences. `sentences.txt` contains the corresponding text sentences. You can see an example on how to load and plot an example sentence in `example.ipynb`. Each handwritten sentence is represented as a 2D array with T rows and 3 columns. T is the number of timesteps. The first column represents whether to interrupt the current stroke (i.e. when the pen is lifted off the paper). The second and third columns represent the relative coordinates of the new point with respect to the last point. Please have a look at the plot_stroke if you want to understand how to plot this sequence.

### Task 1: Unconditional generation.

### Task 2: Conditional generation.

