# Modifying...



## Kyuye Song's_Machine Learning Camp in Jeju, 2017 
# Relational GAN (REGAN) : Generating affective sentence using Relation Networks


## Base theory : Relation Network published by DeepMind, 2017
<li><a href="https://www.python.org/downloads/release/python-2712">Relational Reasoning </li>
is learning to understand relations between different objects(ideas). This is considered an essential characteristic of intelligence.To this end, Relation Networks (RNs) are proposed to solve problems hinging on inherently relational concepts. To be more specific, RN is a composite function:

<img src="https://github.com/gitlimlab/Relation-Network-Tensorflow/blob/master/figure/rn_eq.png", width="700">


The Relational Network for O (O is the set of objects you want to learn relations of) is a function fɸ.
gθ is another function that takes two objects :oi , and oj. The output of gθ is the ‘relation’ that we are concerned about.
Σ i,j means , calculate gθ for all possible pairs of objects, and then sum them up.
where o represents inidividual object while f and g are functions dealing with relational reasoning which are implemented as MLPs. Note that objects mentioned here are not necessary to be real objects; instead, they could consist of the background, particular physical objects, textures, conjunctions of physical objects, etc. In the implementation, objects are defined by convoluted features. 


[bAbi task using Relation Network]
<li><a href="https://github.com/facebook/bAbI-tasks"> bAbi Dataset</li>

1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway 4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2



## Model : Relational GAN (REGAN) to generate affective sentnece





### Prerequisites

<ul>
<li><a href="https://www.python.org/downloads/release/python-2712">Python 2.7.12</li>
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

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

