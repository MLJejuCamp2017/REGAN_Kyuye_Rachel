### Kyuye Song's_Machine Learning Camp in Jeju, 2017 
# Relational GAN (REGAN) : Generating affective sentence using Relation Networks


### Base theory : Relation Network published by DeepMind, 2017
[Relational Reasoning](https://www.python.org/downloads/release/python-2712) is learning to understand relations between different objects(ideas). This is considered an essential characteristic of intelligence.To this end, Relation Networks (RNs) are proposed to solve problems hinging on inherently relational concepts. To be more specific, RN is a composite function:

<ul><img src="https://github.com/gitlimlab/Relation-Network-Tensorflow/blob/master/figure/rn_eq.png", width=300, align="middle"></ul>


The Relational Network for O (O is the set of objects you want to learn relations of) is a function fɸ.
gθ is another function that takes two objects :oi , and oj. The output of gθ is the ‘relation’ that we are concerned about.
Σ i,j means , calculate gθ for all possible pairs of objects, and then sum them up.
where o represents inidividual object while f and g are functions dealing with relational reasoning which are implemented as MLPs. 

[Example : [bAbi task](https://github.com/facebook/bAbI-tasks) using Relation Network] 

1 Mary moved to the bathroom

2 John went to the hallway.

3 Where is Mary?        bathroom        1


### Model : Relational GAN (REGAN) to generate affective sentnece

- Motivation
- Expected output,goal
- Contributions
- How (Scenario)
- Conclusions


### Prerequisites

<ul>
<li><a href="https://www.python.org/downloads/release/python-2712">Python 2.7.12</a></li>
<li><a href="https://github.com/tensorflow/tensorflow/tree/r1.0">Tensorflow 1.0.0</a></li>
<li><a href="http://www.numpy.org/">NumPy</a></li>
<li><a href="https://github.com/pandas-dev/pandas">pandas(powerful data analysis,manipulation library for Python)</a></li>
<li><a href="https://docs.python.org/3/library/json.html">json</a></li>
<li><a href="https://docs.python.org/2/library/re.html">re(Regular expression operations)</a></li>
<li><a href="https://matplotlib.org/">matplotlib</a></li>
</ul>

```
import pandads
import tensorflow as tf 
```
### Git Struture
1. DataSet : all types of dataset 
2. REGAN : code of main model 
3. cloudML : code to train in cloud ML Engine enviroment
4. local : code to train local Engine enviroment
5. 230 seminar ppt: Cognitive science behind ML_Kyuye 07.07.pdf
6. README.md



### Installing

1. Dataset 

    1. raw data <p>
    
      - twitter dataset with emotion tag : twitter_emotion_v2(p,n,N).csv
      - twitter dataset with emotion tag : twitter_emotion_v2(p,n,N).txt
    <p>
    <p>
    
    2. targeted emotion all sentence among dataset 
    
      - Positive emotion : Positive.tsv
      - Negative emotion : Negative.tsv
      - Neutral emotion : Neutral.tsv
    <p>
    <p>
    
    3. tageted emotion one sentence among dataset 
    
      - Positive sentence :<p>
      *men always remember love because of romance only The best love is the kind that awaken the soul that makes us reach for        more that plants the fire in our hearts and brings peace to our minds That's what I hope to give you forever The 
      greatest happiness of life is the declaration that we are loved loved for myself or rather loved in hurt of myself The 
      best and most beautiful things in this world cannot be seen or even heard but must be felt with the heart*
     
     - Negative sentence :<p>
      *My sadness has become an addiction when i am not sad i feel lost I start to panic trying to find my way back which leads       me back to my original state You were rarely wishing for the end of pain the monster said your own pain end to how it you 
      It is the most human wish of all everyone in life is gonna hurt you you just have to figure out which people are worth 
      the pain The World is mad and the people are sad The saddest thing is when you are feeling real down you look around and 
      realize that there is no shoulder for you I guess that is what saying goodbye is always like jumping off an edge The    
      worst art is making the choice to do it Once you are in the air there is nothing you can do but let go*
   
    - Neutral sentence : <p>
     *You cannot visit the past but thanks to modern photography you can try to create it Just ask I was a student at a school 
     and picture her travel across returned to the site exactly 30 years later The picture decided to create some of her 
     favorite picture from back in the day I thought it would be a fun picture project for my YouTube channel tells I was 
     amazed at how little these places had changed Before she left he finish out her old photo albums and scan favorite images 
     Once in she successful track down the exact locations and follow her pose from 30 years previous creating new versions of 
     her favorite she has showed the then and now picture on her YouTube*
  
    <p>
    <p>
     4. train dataset 
      - training output : train_set.txt
      - embeddings(key:value) : word2vec_map.json
    
    
2. Model code
  
    1. model.py
    2. data_loader.py
    3. preprocessing.py
    4. ustils.py 
    5. word2vec.py

```
[raw data]
twitter_emotion_v2(p,n,N).csv

tweet_id,Sentiment,author,content,,,,,,,
1956967341,Neg,xoshayzers,@tiffanylue i know  i was listenin to bad habit earlier and i started freakin at his part =[,,,,,,,
1956967666,Neg,wannamama,Layin n bed with a headache  ughhhh...waitin on your call...,,,,,,,
1956967696,Neg,coolfunky,Funeral ceremony...gloomy friday...,,,,,,,
1956967789,Pos,czareaquino,wants to hang out with friends SOON!,,,,,,,
1956968416,neutral,xkilljoyx,"@dannycastillo We want to trade with someone who has Houston tickets, but no one will.",,,,,,,
....
```
```
[targeted emotion all sentence among dataset]
Negative.tsv

i know i was listenin to bad habit earlier and i started freakin at his part
Layin n bed with a headache ughhhh...waitin on your call...
Funeral ceremony...gloomy friday...
Re-pinging why didn't you go to prom? BC my bf didn't like my friends
I should be sleep, but im not! thinking about an old friend who I want. but he's married now. damn, he wants me 2! scandalous!
Hmmm. is down
Charlene my love. I miss you
I'm sorry at least it's Friday?
..
```
```
[train dataset]
train_set.csv

7,7
7,51
7,22
7,4642
22,51
..
```

```
word2vec_map.json

{"*clutches": [-0.40713953971862793, -0.7928721904754639, 0.5400254726409912, -0.7113916873931885, -0.2588541507720947, 0.31255054473876953, 0.3822500705718994, -0.8085153102874756, -0.25881052017211914, -0.8527586460113525, 0.8659896850585938, 0.3053250312805176, 0.06038784980773926, -0.9497530460357666, 0.4123997688293457, 0.5235145092010498, 0.7868549823760986, 0.37369585037231445, -0.8102536201477051, -0.5631704330444336, 0.7328002452850342,.....]}

```

### Running 

1. Experiment 1 : generating sentence based on 3 emotion label(positive,negative,netural)
    - 3emo_raw.sh

2. Experiment 2-1 : generating sentence based on 1 emotion all dataset 
    1. Positive : 1emo_pos.sh
    2. Negative : 1emo_neg.sh
    3. Neutral  : 1emo_neu.sh
  
3. Experiment 2-2 : genetrating sentence based on 1 emotion only one sentence 
    1. Positive : 1sent_pos.sh
    2. Negative : 1sent_neg.sh
    3. Neutral  : 1sent_neu.sh
  

```
./1emo_pos.sh

```



### Result (Modifying)

1. Experiment 1 : generating sentence based on 3 emotion label(positive,negative,netural)
2. Experiment 2-1 : generating sentence based on 1 emotion all dataset 
    1. Positive
    2. Negative
    3. Neutral
  
3. Experiment 2-2 : genetrating sentence based on 1 emotion only one sentence 
    1. Positive
    2. Negative
    3. Neutral

```
Give an example
```


## Built With

* [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) - Visualizing Learning
* [CloudML](https://cloud.google.com/ml-engine/) - Managed Scalable Machine Learning 


- ## Contributing

Please read [READEME.mg](https://github.com/MLJejuCamp2017/REGAN_Kyuye_Rachel) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Kyuye Song(Rahcel)** - *Initial work* - [PurpleBooth](https://github.com/kyuye)
* **Chanwoo Lee(Jacob)** - *Initial work* - [PurpleBooth](https://github.com/leechanwoo)


## Acknowledgments
* Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap,[A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427)
* Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush, Bart van Merriënboer, ["Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"](https://arxiv.org/abs/1502.05698)
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, ["End-To-End Memory Networks"](https://arxiv.org/abs/1503.08895)


