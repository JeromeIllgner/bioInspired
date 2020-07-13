# Setup and Usage
Install Pipenv to manage dependencies
```
pip install pipenv
```

Add pipenv to your path and run the program with:
```
pipenv run network --help
```

# Biologically Inspired Computation (F20BC/F21BC) 
# Coursework 
**This assessment is worth 50% of your course mark**

**Due: 3:30pm, Monday 25th November 2019 (Week 11)**
## Overview 
This assessment aims to increase your understanding of both artificial neural networks (ANNs) and particle swarm 
optimisation (PSO), two biologically-inspired techniques which are covered in the course. It involves implementing 
both an ANN and PSO from scratch, and experimentally investigating how PSO can be used to optimise an ANN to carry out a
specified task. Please read through the following important points before you begin: 
- In order to encourage discussion of the topic, **the assessment involves working in pairs (i.e. 2 people)**. 
Please choose your own partner, and inform us of this choice by the end of week 3. You must choose one partner, no 
collaboration beyond that is allowed. You should communicate your partner name to Dr Patricia A. Vargas at most by the 
03/10/2019 (Thursday) via email, with the subject: “F20/F21BC 2019 – CW PARTNER” including both **names** and 
**surnames** and **student ID**. No changes will be allowed after that date. 
- If you are unable to find a partner, please let us know, and we will attempt to match you up with another student. 
Each member of a pair should contribute equally, and we will be asking you to assess the contribution of your peer 
when you hand-in the work. 
- **You do not need to wait until we have covered all the related topics in the lectures to start working on your 
project** . ANNs will be covered in the first half of the course. PSO will be covered in the first week (Week 7) of the 
second part of the course. 
- This assessment involves programming. If you have no previous experience of programming, 
and do not feel able to contribute usefully to this assessment, then please come and talk to 
your lecturer(s) **as soon as possible**  about doing an alternative assessment. 

**This is assessed coursework** . You are allowed to discuss this assignment with students outside of your pair, 
>but you should not copy their work, and you should not share your own work with other students. We will be carrying
> out automated plagiarism checks on both code and text submissions. **Special note for re-using existing code.**  
>If you are re-using code that you have not yourself written, then this must clearly be indicated, making clear which 
>parts were not written by you and clearly stating where it was taken from. If your code is found elsewhere by the 
>person marking your work, and you have not mentioned this, you may find yourself having to go before a disciplinary 
>committee and face grave consequences. **Late submission and extensions.**  Late submissions will be marked according 
>to the university's late submissions policy, i.e. a 30% deduction if submitted within 5 working days of the 
>deadline, and a mark of 0% a^er that. The deadline for this work is not negotiable. If you are unable to complete the 
>assignment by the deadline due to circumstances beyond your control (e.g. illness or family bereavement), you should 
>complete and submit a mitigating circumstances application: 
[mitigating circumstances form](http://www.hw.ac.uk/students/studies/examinations/mitigating-circumstances.htm)

### Detailed Description 
What you are asked to do: 
1. Implement a	multi-layer ANN architecture 
2. Implement a	PSO algorithm 
3. Use the PSO algorithm to optimise the ANN’s parameters 
4. Investigate how hyperparameters effect the ability of PSO to optimise ANNs 
5. Write a	report and submit both this and your code to VISION 6. Sign the “Coursework Group Signing Sheet” and 
hand it in together with your report. 

These tasks are described in more detail below. Implementation should be done using a	language of your choice 
(e.g. Java, Python, Matlab, C, C++). The aim is for you to learn how to implement biologically-inspired approaches from 
scratch, so you should **not**  use existing ANN or PSO libraries. 
#### 1. Implement a	multi-layer ANN architecture 
You should implement a	simple feedforward multilayer architecture. Both the number of neurons in each layer and the 
number of layers should be configurable. You do not need to implement any classical training algorithms, e.g. 
backpropagation, since you will be using PSO to do the training. Here is the list of activation functions that should 
be used, though you may also investigate other suitable functions:

| Number | Activation Function Name | Equation  |
|--------|--------------------------|-----------|
| 0 | Null                          | `0`               |
| 1 | Sigmoid                       | `1 / (1 + exp(-x)`|
| 2 | Hyperbolic Tangent            | `tanh(x)`         |
| 3 | Cosine                        | `cos(x)`          |
| 4 | Gaussian                      | `exp(-(x^2)/2)`   |

#### 2. Implement a	PSO algorithm 
You should implement a	PSO algorithm. You might find the pseudocode in the book “Essentials of Metaheuristics” 
useful for this task: [link to book](https://cs.gmu.edu/~sean/book/metaheuristics/)

Here are some points to bear in mind: 
- Each particle should have a	group of informants (i.e. not just the population best). How you choose these is up to 
you; for instance, they could be randomly allocated at initialisation. You might want to compare different strategies 
for allocating informants. 
- There are lots of variants of PSO. Whilst you are not expected to be aware of all of these, 
you are encouraged to read about PSO and experiment with different approaches. 

#### 3. Use the PSO algorithm to optimise the ANN’s parameters 
The problem domain for this work is function approximation, i.e. given a function y = fn(x), find an ANN that 
implements this function. In principle, there exists an ANN that can implement any mathematical function; however, 
the challenge lies in finding it. You will be using PSO to do this. Each group will develop a program to evolve a MLP 
(Multi-Layer Perceptron) artificial neural network for a task of function approximation. The functions to be 
approximated are named: linear, cubic, sine, tanh, xor and complex. Data will be provided separately for each function 
on a .txt file format.

| Function Name | Equation      |
|---------------|---------------|
| linear        | y=x           |
| cubic         | y=x^3         |
| sine          | y=sin(x)      |
| tanh          | y=tanh(x)     |
| xor           | y=x1 xor x2   |
| complex       | y=1.9{1.35 + e^(x1-x2) * sin[13(x1 - 0.6)^2] * sin[7x2]} |

Here are some points to bear in mind:
- Standard PSO represents solutions as a vector of floating-point values. Consequently, the parameters of your ANN’s 
architecture (connection weights, neuron activation functions etc.) need to be encoded as a	vector of floating-point 
values. 
- To evaluate how well an ANN implements a	particular function, you should create a group of (x,y) pairs and measure 
how close the ANN’s output is to y for each value of x. The program should allow you to visualise on running time 
the graphs of:
1) the desired output and the ANN output for each function 
2) the mean squared error
```
(MSE) = 1/n * sum( (d[i] - u[i])^2 )
```

Where:
 - n = max number of samples
 - d = desired output
 - u = output of a	neuron on the output layer

Remember: 
- Each time a solution is evaluated in PSO, the values in the vector should be used to set the parameters of the ANN, 
and the ANN should then be evaluated against the (x,y) pairs from the function that is being approximated. 
- Although there are versions of PSO that can handle variable-length vectors, you are not expected to know about these. 
Consequently, the architecture of the ANN (i.e. the number of layers and neurons) should be specified at the beginning 
of a PSO run and remain fixed. 

#### 4. Investigate how hyperparameters effect the ability of PSO to optimise ANNs 

PSO has a number of hyperparameters (e.g. the population size, the number of informants per particle, the acceleration 
coefficients) and these all have an effect upon the algorithm’s behaviour.

ANN architectures also have hyperparameters (e.g. number of layers, choice of activation functions), and these all 
affect how easy it is to train the ANN to implement a particular behaviour. For this part of the coursework, you should 
carry out an experimental investigation of how these hyperparameters effect the ability of PSO to optimise ANNs for the 
function approximation task. Here are some points to bear in mind:
- You can begin by informally trying out different values for the various hyperparameters and getting a	feel for how
 much they effect the results. 
 - Pick three or four hyperparameters which you think have significant effect and then 
 carry out a more formal experimental investigation. 
 - Pick a sensible range of values for each hyperparameter that you investigate. This could be guided by values you 
 find in books, published papers etc. 
 - PSO is a stochastic algorithm, meaning that for the same problem and hyperparameter values, you will likely get 
 different results each 9me you run it. Therefore, the distribution of results across a series of runs is more 
 informative than the result of a single run. For example, it is common to give the mean result across at least 
 10 repeated runs.
 
#### 5. Write a	report and submit both this and your code to Vision 
Your report should: 
- Be no more than **6 pages** in length (max of 3,000 words). You should take this into account when planning your 
experiments. If you have more results than you have space for, then pick the results that you think are most insightful
 and briefly mention which other experiments you carried out. 
- Briefly describe your implementations of ANN and PSO, noting any interesting aspects. 
- Your report should contain the following sections: 
1. Introduction
2. Program development rationale
3. Methods 
4. Results
5. Discussion and Conclusion 
6. References 
- Report the results of your experimental investigation. For instance, you might use tables that show the average 
results achieved for each function approximation problem, and plots that show how hyperparameter values effect these 
averages. 
- Referring to these results, discuss how the various hyperparameters affected the performance of your implementations 
of ANNs and PSO, and say why you think this is the case. 
- Include useful references to the wider literature. For instance, you might use references to books or papers to 
justify particular implementation choices, or you could compare your findings to those reported elsewhere. 
Use standard referencing styles for this. You should submit both your report (as a **pdf**  file) and your code 
(as a **zip**  file) to Vision using the links provided. Grading will use the assessment criteria given in the 
table below. 

#### 6. Sign the “Coursework Group Signing Sheet” and hand it in together with your 
#### report 
You will find the Coursework Group Signing Sheet paper on VISION under <Assessment>. 
**NOTE: No marks will be issued until we have this signed copy ** 

**Marking scheme for F20BC Students : This assessment scores 50% of the overall course assessment.**

| Criteria | Weight | A (70-100%) | B (60-69%) | C (50-59%) | D (40-49%) | E/F (<40%) |
|----------|--------|-------------|------------|------------|------------|------------|
| Implementation (i.e. code for ANN and PSO, and evaluation, comments and documentation) | 45% | Creative implementations of ANN and PSO that exceed the basic requirements. Correct evaluation code. Easy to read and well structured. | Correct implementations of the basic requirements. Generally good coding, structure and documentation. | Some significant issues in terms of correctness, structure, coding practice and documentation. | Major issues in terms of correctness, structure, coding practice and documentation. | Critical errors: for example, the code does not compile and/or run, or inappropriate algorithms have been implemented. |
| Experimental study (i.e. choice and validity of experiments performed, presentation of results) | 20% | Hyperparameters investigated are well motivated and their values are well chosen. Suitable results have been collected and are clearly presented and meaningful. | Some minor issues in terms of the motivation or description of hyperparameters, the experiments performed, or the presentation of results. | Some significant issues in terms of the motivation or description of hyperparameters, the experiments performed, or the presentation of results. | Some major issues: experiments do not make sense, have invalid results, or the study is not adequately described. | 
| Some critical issues: experimental study is nonsensical or missing, the experiments are inappropriate, or the description of the study is uninformative. Wider discussion (i.e. intro, interpretation of results, conclusions, use of the wider literature) | 20%  | Clear, insightful discussion that shows a	good understanding of ANNs and PSO and includes well chosen references to the wider literature. | Generally clear and insightful, but shows some misunderstanding of ANNs and PSO. Adequate use of the wider literature. | The discussion is limited in terms of the depth or volume of understanding it demonstrates. Little or no use of the wider literature. | Some major issues in terms of depth or volume of understanding. No use of the wider literature. | No real demonstration that the subject matter has been understood, or very limited in its scope. |
| Report (i.e. structure, language, referencing etc.) | 15% | Report is well structured and divided into sections; good use of language; consistent use of font Arial, size 12; perfect use of Harvard referencing style | Report is suitably structured and divided into sections; mostly good use of language; use of font Arial, size 12; use of Harvard referencing style | Report is structured but not divided into sections; language issues that affect readability; inconsistent use of fonts and sizes; mixed use of referencing styles | Report is poorly structured; substantial language issues that affect readability; use of different fonts and sizes; no referencing style | Report has a nonsensical structure; language issues make it very hard to read; use of different fonts and sizes; no referencing style |
