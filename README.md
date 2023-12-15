# telco_customer_churn

# **Introduction**
An organization loses its customers to its competition for various reasons. What mainly attracts customers to go ahead with a shift is attributed to the price of the product and its quality. This is true for the Telecom industry as well. A churn of 10 to 60 percent in any company is a big number that can affect the company’s overall growth.

Churn is the percentage of customers who stop using a company's product or service during a particular period of time. Churn category refers to the reason why a customer stops using a company's service, while churn reason refers to the specific reason for a customer to churn. In the telecom industry, multiclass classification is used to predict a customer's churn category and churn reason.

The main goal of multiclass and multilevel classification in telecom is to predict the churn category and churn reason of a customer, which can help a telecom company to take proactive measures to reduce customer churn, improve customer retention and enhance customer satisfaction. By understanding the reasons behind churn, a telecom company can develop targeted retention strategies to retain customers.
    
### **Business Impact of Churn Classification**



**Increase revenue with customer retention**: Reduced churn means company is not losing out against it competitors and a happy customer would keep on spending money on the platform


**Improve customer acquisition cost**: If the company is able to stop old customers from leaving then it doesn't need to spend extra money on getting new customers or in an extreme case throw offers to get the old customer back. This impacts cost of acquisition per customer


**Improve customer satisfaction**: If the company is able to identify which customer will churn and reason behind that, it can fix the problem and in the end improve customer satisfaction

# Machine learning:

Machine learning systems combines strength of historical data and statistical techniques to explain the right churn reason for individual customers of the company. For example, a ML system can tell with a high confidence if a potentail customer will churn or not and what is the reason for the churn.

### **Assumptions**

* We assume that <b>Churn Category & Churn Reason</b> are our target variables.
* We assume that whenever the churn reason is attributed to wrong churn category, it should be switched to the proper one.
* The primary goal is that of Churn category identificationa multi-class classification.
* Post that we incorporate churn reason and formulate the problem as a multi-label classification problem.
* <i>Churn Category = Not Applicable</i> denotes that the customer didn't churn and it is our negative label.
* Columns with a lot of null values are not meaningful and imputation also won't be helpful.
* Most of the missing values are imputed(some are remaining as they can be imputed according to the problem).

## **Approach**


We are treating this problem as a supervised learning problem doing multi-label classification. So every data point will have multiple target variables(in this case it would be 2, i.e., churn category and churn reason) for the model to learn the dependencies and predict on the unknown.


In real life, this model would tell the business that which category of churn does a user lie in and the reason behind the churn. It would in turn help the company to proactively prevent customers from leaving the platform.


Given our assumptions about the data, we will build a prediction model based on the historical data. Simplifying, here's the logic of what we'll build:


1. We will try to understand the churn data first and do a detailed problem specific EDA;
2. We'll build a model to predict the churn category using multi-class classification;
3. We'll then formulate the problem as a multi-label classification problem and predict both churn category and churn reason

**Supervised Machine Learning:**

In supervised machine learning, the algorithm is trained on an labeled dataset with a predefined target variable. The goal is to identify patterns, relationships, and structures of the data with the target variable, such as logistic regression, decision tree or boosting trees


**Multi-Class Classification:**

Multiclass classification is a type of machine learning task where the goal is to classify instances into one of three or more classes. In other words, given a set of input data, the task is to predict the class label of the instance, where the instance can belong to any one of the several predefined classes.

For example, in a medical diagnosis application, the goal may be to predict whether a patient has a certain disease, where the disease can be one of several possibilities. In this case, the output of the classification model would be a probability distribution over the possible classes, with the highest probability indicating the predicted class.

There are various algorithms that can be used for multiclass classification, including logistic regression, decision trees, random forests, support vector machines, and neural networks. The choice of algorithm depends on various factors, such as the nature of the data, the number of classes, and the available computing resources.

**Multi-Label Classification:**

Multilabel classification is a type of machine learning task where the goal is to assign one or more class labels to an instance. Unlike multiclass classification, where an instance can belong to only one class, in multilabel classification, an instance can belong to more than one class simultaneously.

For example, in a news article categorization task, an article may belong to multiple categories, such as "politics", "world news", and "entertainment". In this case, the task would be to predict a set of labels that best describe the content of the article.

Multilabel classification is used in a variety of applications, such as text classification, image annotation, and recommendation systems, where multiple labels can be assigned to an instance based on its content or characteristics.

## **Data Dictionary**



| Column name	 | Description|
| ----- | ----- |
| Customer ID|  Unique identifier for each customer |
| Month|  Calender Month- 1:12 |
| Month of Joining|  "Calender Month -1:14|   Month for which the data is captured" |
| zip_code|  Zip Code |
| Gender|  Gender |
| Age|  Age(Years) |
| Married|  Marital Status |
| Dependents|  Dependents - Binary |
| Number of Dependents|  Number of Dependents |
| Location ID|  Location ID |
| Service ID|  Service ID |
| state|  State |
| county|  County |
| timezone|  Timezone |
| area_codes|  Area Code |
| country|  Country |
| latitude|  Latitude |
| longitude|  Longitude |
| arpu|  Average revenue per user |
| roam_ic|  Roaming incoming calls in minutes |
| roam_og|  Roaming outgoing calls in minutes |
| loc_og_t2t|  Local outgoing calls within same network in minutes |
| loc_og_t2m|  Local outgoing calls outside network in minutes(outside same + partner network) |
| loc_og_t2f|  Local outgoing calls with Partner network in minutes |
| loc_og_t2c|  Local outgoing calls with Call Center in minutes |
| std_og_t2t|  STD outgoing calls within same network in minutes |
| std_og_t2m|  STD outgoing calls outside network in minutes(outside same + partner network) |
| std_og_t2f|  STD outgoing calls with Partner network in minutes |
| std_og_t2c|  STD outgoing calls with Call Center in minutes |
| isd_og|  ISD Outgoing calls |
| spl_og|  Special Outgoing calls |
| og_others|  Other Outgoing Calls |
| loc_ic_t2t|  Local incoming calls within same network in minutes |
| loc_ic_t2m|  Local incoming calls outside network in minutes(outside same + partner network) |
| loc_ic_t2f|  Local incoming calls with Partner network in minutes |
| std_ic_t2t|  STD incoming calls within same network in minutes |
| std_ic_t2m|  STD incoming calls outside network in minutes(outside same + partner network) |
| std_ic_t2f|  STD incoming calls with Partner network in minutes |
| std_ic_t2o|  STD incoming calls operators other networks in minutes |
| spl_ic|  Special Incoming calls in minutes |
| isd_ic|  ISD Incoming calls in minutes |
| ic_others|  Other Incoming Calls |
| total_rech_amt|  Total Recharge Amount in Local Currency |
| total_rech_data|  Total Recharge Amount for Data in Local Currency |
| vol_4g|  4G Internet Used in GB |
| vol_5g|  5G Internet used in GB |
| arpu_5g|  Average revenue per user over 5G network |
| arpu_4g|  Average revenue per user over 4G network |
| night_pck_user|  Is Night Pack User(Specific Scheme) |
| fb_user|  Social Networking scheme |
| aug_vbc_5g|  Volume Based cost for 5G network (outside the scheme paid based on extra usage) |
| offer|  Offer Given to User |
| Referred a Friend|  Referred a Friend : Binary |
| Number of Referrals|  Number of Referrals |
| Phone Service|  Phone Service: Binary |
| Multiple Lines|  Multiple Lines for phone service: Binary |
| Internet Service|  Internet Service: Binary |
| Internet Type|  Internet Type |
| Streaming Data Consumption|  Streaming Data Consumption |
| Online Security|  Online Security |
| Online Backup|  Online Backup |
| Device Protection Plan|  Device Protection Plan |
| Premium Tech Support|  Premium Tech Support |
| Streaming TV|  Streaming TV |
| Streaming Movies|  Streaming Movies |
| Streaming Music|  Streaming Music |
| Unlimited Data|  Unlimited Data |
| Payment Method|  Payment Method |
| Status ID|  Status ID |
| Satisfaction Score|  Satisfaction Score |
| Churn Category|  Churn Category |
| Churn Reason|  Churn Reason |
| Customer Status|  Customer Status |
| Churn Value|  Binary Churn Value |

To make the lattitudinal and longitudinal data more clear for visualizations we will cluster the data into hexagons and plot the information on a map. We can then determine the number of customers and the percentage of churn customers by hexagon. It can help us assess customer density across an even area. We do not want to delve into the analysis of each individual city, so we will use a fairly large hexagon size.

### What were the reasons for the churn mentioned by customers?¶

Now we can start analyzing our customer profile data to understand which type of customers are more likely to stop using our service and what actions we can take.

But before that, we can also look at the customer responses that we have in the Churn Reason column. Of course, customer survey data is often biased, as it contains the subjective opinion of the customer, but asking the opinion of customers is very important for any business that wants to develop and improve. Moreover, we have a lot of information with which we can verify customer responses.

**Observation**
* 6.2% of churn customers left the service due to service dissatisfaction
* Another 6.1% of churn customers left the service due to Lack of self-service on Website
* We also have a number of reasons for the churn that we cannot understand in any way
* The median charges of customers who have gone into churn are higher than the median charges of customers who use the service
* But this does not necessarily mean that customers who stopped using the service were more affluent. After all, we have already learned that many customers leave the service in the first 2 months of using the service

# **Data Processing & Feature engineering**

### Dropping Irrelevant Features and IDs

* Location ID
* Service ID
* area_codes
* Status ID
* Customer ID
* zip_code
* state
* county
* latitude
* longitude', 
* 'night_pck_user
* fb_user
* Customer Status'


## SMOTE

* We have a highly imbalanced dataset
* We should use over and under sampling to make our dataset more suited for the ML model

SMOTE (Synthetic Minority Over-sampling Technique) is a technique used in machine learning to address class imbalance in a dataset. Class imbalance occurs when the number of instances in one class is much lower than the number of instances in another class, making it difficult for machine learning algorithms to learn from the data and predict the minority class accurately.

SMOTE works by creating synthetic samples from the minority class by interpolating new instances between existing instances. The new instances are created by selecting pairs of instances that are close to each other in the feature space and generating new instances along the line that connects them. The number of new instances to be generated is determined by a user-defined parameter that specifies the desired ratio of minority to majority class instances.

The synthetic instances generated by SMOTE are used to balance the classes in the dataset, allowing the machine learning algorithm to learn from a more balanced dataset and make better predictions on the minority class.

## Undersampling

Undersampling is a technique used in machine learning to address class imbalance in a dataset. Class imbalance occurs when the number of instances in one class is much lower than the number of instances in another class, making it difficult for machine learning algorithms to learn from the data and predict the minority class accurately.

Undersampling works by randomly selecting a subset of instances from the majority class so that the number of instances in the majority class is reduced to a level comparable to the number of instances in the minority class. This creates a more balanced dataset and allows the machine learning algorithm to learn from a more representative sample of the data.

Undersampling can be effective in reducing the computational cost and training time of machine learning models, as well as reducing the risk of overfitting to the majority class.

## **Supervised learning**



Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.

Supervised learning can be separated into two types of problems when data mining—classification and regression:

1. Classification uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.


2. Regression is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms.

## **Decision Trees**

**Decision Trees in Classification**

Decision trees are a type of supervised learning algorithm that can be used for classification as well as regression problems. They are widely used in machine learning because they are easy to understand and interpret, and can handle both categorical and numerical data. The idea behind decision trees is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.


**Splitting Criteria**

To build a decision tree, we need a measure that determines how to split the data at each node. The splitting criterion is chosen based on the type of data and the nature of the problem. The most common splitting criteria are:

* Gini index: measures the impurity of a set of labels. It calculates the probability of misclassifying a randomly chosen element from the set, and is used to minimize misclassification errors.
* Information gain: measures the reduction in entropy (uncertainty) after a split. It is used to maximize the information gain in each split.
* Chi-square: measures the difference between observed and expected frequencies of the classes. It is used to minimize the deviation between the observed and expected class distribution.

**Ensemble Methods**

Ensemble methods are techniques that combine multiple models to improve performance and reduce overfitting, a typical issue with decision trees. The two most common ensemble methods used with decision trees are:

* Bagging 

We will be using Random Forest as a bagging technique to introduce bootstrap sampling is a statistical technique that involves randomly sampling the data with replacement to create multiple subsets. These subsets are used to train individual decision trees. By using bootstrap samples, the algorithm can generate multiple versions of the same dataset with slightly different distributions. This introduces randomness into the training process, which helps to reduce overfitting over Bagging techniques for algorithms.

* XGBoost

We will be using XGBoost as a the key idea behind XGBoost is that it improves upon the predictions of the weak learners by focusing on the misclassified data points. By fitting a new tree to the residuals, XGBoost can correct the errors of the previous model and improve its overall accuracy. Additionally, XGBoost uses regularization to prevent overfitting and to improve generalization performance.

Deecision trees are powerful tools for classification problems that provide a clear and interpretable representation of the decision rules learned from the data. The choice of splitting criterion, stopping criterion, and ensemble method can have a significant impact on the performance and generalization of the model.

### **Gradient Boosting**

The primary idea behind this technique is to develop models in a sequential manner, with each model attempting to reduce the mistakes of the previous model.The additive model, loss function, and a weak learner are the three fundamental components of Gradient Boosting.

The method provides a direct interpretation of boosting in terms of numerical optimization of the loss function using Gradient Descent. We employ Gradient Boosting Regressor when the target column is continuous, and Gradient Boosting Classifier when the task is a classification problem. The "Loss function" is the only difference between the two. The goal is to use gradient descent to reduce this loss function by adding weak learners. Because it is based on loss functions, for regression problems, Mean squared error (MSE) will be used, and  for classification problems, log-likelihood.

### **XG Boost**




To improve upon the first decision tree, we can use XGBoost. Here's a roadmap for it:

* Initialize the model: We start by initializing the XGBoost model with default hyperparameters. This model will be a simple decision tree with a single split.

* Make predictions: We use this model to make predictions on the training data. We compare these predictions to the true labels and calculate the residuals, which are the differences between the predicted values and the true labels.

* Fit a new tree: We then fit a new decision tree to the residuals. This tree will be a weak learner, as it is only modeling the errors of the previous model.

* Combine the models: We add the new tree to the previous model to create a new ensemble. This new ensemble consists of the previous model plus the new tree.

* Repeat: We repeat steps 2-4 for a specified number of iterations, adding a new tree to the ensemble each time.

* Predictions: To make predictions on new data, we combine the predictions of all the trees in the ensemble.

## Classification Evaluation Metrics

#### The following evaluation metrics have been identified as important for addressing the business problem:

**F1 score:** Use the F1 score when the class distribution is imbalanced, and when both precision and recall are equally important.

**Recall score:** The recall score will be used as the cost of false negatives (missing customers likely to churn) is high. For example, in this project missing out on customers that are likely to churn is more important than misclassifying customers that arne't at risk of churn.

**Confusion matrix:**  The confusion matrix is a versatile tool that can be used to visualize the performance of a model across different classes. It can be useful for identifying specific areas of the model that need improvement. As this project will be used in an iterative manner, it will be important to optimize the model by analyzing previous models' failures.

**ROC AUC score:** the ROC AUC score will be used as it's ability to distinguish between positive and negative classes is important. Ideally we would like to have the clearest picture possible in terms of a customer's likelihood of churn so as to not needlessly waste resources on customers that don't churn and properly identify customers with a good risk of churn.

## Categorical Accuracy

Right now there are errors in our accuracy score, we will need to revise our Deep Neural Network model or the data set to improve performance.