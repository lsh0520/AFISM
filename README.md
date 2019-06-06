# Adversarial Personalized Ranking for Recommendation

APR enhances the pairwise ranking method BPR by performing adversarial training. To illustrate how it works,  APR on FISM is implemented here by adding adversarial perturbations on the matrices P and Q.

This is the implementation for Sihang Li's graduation thesis @USTC, under the supervision of Prof. Xiangnan He:


## Environment

Python 2.7

TensorFlow  1.5


PS. For your reference, our server environment is Intel(R) Xeon(R) Gold5118CPU@2.30GHz and 64 GiB memory. We recommend your free memory is more than 64 GiB to reproduce our experiments( 96GiB is preferred when it comes to Gowalla dataset).


## Dataset
We use two processed datasets:  Yelp(yelp), Gowalla(gowalla) in Data/

**train.rating:**

- Train file.


- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

**test.rating:**

- Test file (positive instances).
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

**test.negative**

- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

*PS. In our experiments, we adopt the **all ranking** evaluation strategy. 
