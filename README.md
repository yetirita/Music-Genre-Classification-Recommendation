# Music-Genre-Classification-Recommendation
Music Genre Classification &amp; Recommendation (MSBD5001)

This is the group project of class MSBD5001

Dataset is from http://marsyas.info/downloads/datasets.html

## Introduction
Music is everywhere. But using ears to understand music is not the only way.
In our project, we visualize the music, extract features as CSV files from the audio. Then, we use the extracted data to classify genres using multiple models and build a recommendation system.

This dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, including blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock,
 each represented by 100 tracks.

Frequency spectrum analysis of audio files is a basic data processing project, and it also prepares data for subsequent feature analysis. Let me briefly introduce these features.
First, we import a 30s audio file. It returns 2 parameters, which we mainly used in feature extraction.
This plot visualizes the amplitude envelope of a waveform. Then, we do short-time Fourier transformation of speech signal and visualize the spectrogram. 

In order to better explore the audio details, we first divide the raw audio into ten 3s audios with the same label. Next, we perform feature extraction both on 30s and 3s audio files.

## Classification
### CNN 
Music classification is mainly composed of three parts. Now let’s come to the first part, Convolutional Neural Network. 

For one piece of music, it can be described by a bunch of cepstrum vectors. Each vector is the MFCC feature vector of one frame. And the number of vectors is mainly based on the sample rate, hop length and the duration of music. Their relationship is roughly like this formula:
Sample rate and hop length is always fixed in the analyses of music, so the number of the vectors depends on music duration. As for the length of one vector, it can be specified manually. 
In our project, all of the music is divided into three-second segments. And for one segment, we can get 130 MFCC vectors. And we extracted 20 coefficients for each frame. 
Stacking all the vectors, we can get a spectrogram feature matrix of size 130 by 20. And this matrix can be used as the input.

The accuracy of validation-set and test-set is nearly 80%, which is relatively high. Display the results in a confusion matrix. The diagonal is distinct. Overall, the result is good
However, if we reduce the sampling rate and generate the 30s songs into the mfcc vectors with the same input size as 3s. The performance of this model is not very good, the accuracy is only about 12%. In other words, the effect of our model has limitations in the length of the original music.

### ML
We split the data into training and testing based on different class labels. We randomly split the train size to 75% and the test size to 25%. 
Then, we implement transformer to standardize features.

Since most of our features are difficult to interpret, we tried PCA to reduce the dimension. And we set the number of principal components equal to 20.  
And we can see the first component and second component through the graph
The variance become larger with the increasing of principle components. 
And we can find it doesn’t work well.

Then, we tried T-Distribution Stochastic Neighbor Embedding.

Because our datasets are balanced, the performance for the model is using the accuracy. We tried 10 models including Logistic Regression, KNN, Decision Tree, etc. 
We use GridSearchCV for parameter tuning. 

After tuning, we find out that the highest accuracy of all models is still lower than 80%. 
We think that is because the datasets are small which means we don’t have enough data for our models. Therefore, we split the songs into 3 seconds. This way increases 10 times the amount of data. Within more data, we expect our models will have better performance. 

The orange column is the accuracy score we use new training model to test 3 seconds of other audio files. The gray column is the accuracy score we use new training model to test 30 seconds of audio files. 
Then we observed that it’s easy to classify the songs with high accuracy scores.

## Generalization
However, we doubted the extremely high accuracy. 
As we mentioned before, we split the data randomly. Some pieces of one song may be split into training set and other parts are in test set. Therefore, some pieces of song have been visited and some information has been gained during training. As a result, our models can accurately classify the songs.
To deal with this problem, we purposely select 75% of 30s songs to split into 3s segments. Those songs are our training set and the others are test set. Then, we can make sure the songs in our training set are complete, no information was learned when training. 

Although CNN is not the best in the previous part, it has the best generalization ability. However, CNN cannot be used to test 30s songs, and the other models cannot perform well. 

In order to solve this problem and generalize our models to songs of different lengths, we used a voting strategy as this diagram shows, a 30s song will get 10 output results, the predictions of these 3s segments will vote on the class of the song. The final genre is the most votes.

The best one is CNN, which has 64% accuracy. 
In general, traditional machine learning method and CNN have their own advantages. CNN has better generalization ability, but machine learning can perform better if we have enough information from training set.
Although the accuracy is not very high. It is meaningful for a 10-class classification with generalization ability.


## Recommendation
### Naïve KNN recommendation
we use Ball Tree to store our Music library and accelerate our KNN query. 
The first part is Naïve KNN recommendation. we first do normalization for our dataset, and simply use the dataset to build a Ball Tree. We try both 3 seconds and 30 senonds segment and we want to find top 5 neighbors and 7 neighbors.
From the table above we can see that the top 5 recommendation result of 3 seconds segment is around 80%, and result of 30s segment is around 60%. For example, we pick two of the query results.

### classification-based recommendation
After doing the classification, we have a label for each music, so we can divide the whole dataset into 10 parts according to the label. For each part, we construct a Ball Tree separately. The process is similar with the Naïve KNN recommendation part.

However, there is a problem. How we evaluate this recommendation model? it is difficult to evaluate the result because we find the neighbors in the music of the same label. Then we find a solution, we use 3s segments to make recommendations and see if the nearest music comes from the same 30s segment.

We can find the accuracy is around 40%, not very high. We think the reason for this situation is the same as mentioned above, because there are many changes in the music.
For example, in the left picture, the results looks well. But in the right picture, some different music appeared in the results. We try to listen to these segments several times, we found that these clips are indeed very similar, which gives us some confidence in the recommendation results.

In these two parts, classification and recommendation, we also have more work to do, we want to combine the advantages of ML and CNN for music classification and we want to find more features and better evaluation method of recommendation.
