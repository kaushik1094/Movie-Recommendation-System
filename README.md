<h1>Movie-Recommendation-System</h1>
<p>Our Deep Learning Model will predict which user is going to like  which movie using the recommendation system using Boltzmann Machine
So we are going to build this two models with Restricted Boltzmann machines and Auto encoders and we are going to calculate the train loss and test loss
we are going to use average distance to evalute our RBM. We obtained the loss of 0.25 which means we have obtained 75% of accuracy. I would like other developers to run it on the 1 million data set provided and test the code via pull request. I am open to modifications of code for improvements</p>
<strong>If you want to check that 0.25 corresponds to 75% of success, you can run the following test:</strong>
                    
                    
                    
                    import numpy as np
                    u = np.random.choice([0,1], 100000)
                    v = np.random.choice([0,1], 100000)
                    u[:50000] = v[:50000]
                    sum(u==v)/float(len(u)) # -> you get 0.75
                    np.mean(np.abs(u-v)) # -> you get 0.25




<p>I dive into dataset and we preprocessed the dataset into a list of users and user ratings for different movies. we converted training set and test into
torch tensors. </p>
<p>I have built a class of RBM using using pytorch libraries and created a RBM class with contrastive divergence to approximate gradient and update weights</p>
<p>Dataset Courtesy: <a href ="https://grouplens.org/datasets/movielens/">Dataset Courtesy </a></p>
<strong>Additional Resources:</strong>
<p><a href="http://pytorch.org/docs/0.3.1/">Pytorch Documentation</a></p>
<p><a href = "http://image.diku.dk/igel/paper/AItRBM-proof.pdf">Introduction to Restricted Boltzmann Machines</a></p>
