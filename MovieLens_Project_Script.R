############ META ##############
# File name   : MovieLens_Project_Script.R
# Author      : Rahma Ali
# Email       : rahma.diab@gmail.com
# Date        : Jan 2020
# Description : A script file part of the MovieLens machine learning project. The code 
# aims to explore and create a movie recommendation system using the MovieLens data. 
# The file is submitted in partial fulfillment of the requirements for obtaining 
# HarvardX Professional Certificate in Data Science. 
################################

################################
# Preamble
################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("data.table", repos = "http://cran.us.r-project.org")

################################
# Download MovieLens data
################################
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

################################
# Create edx set, validation set
################################
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Explore edx data
################################
str(edx)
n_distinct(edx$movieId)
n_distinct(edx$userId)

################################
# Explre movie effect on rating variability
################################
# Distribution of movie ratings
edx %>% group_by(movieId) %>%
  summarize(count=n(), avg_rating=mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(fill="deepskyblue2") +
  ggtitle("Movie Ratings Distribution")


# Scatter plot of average movie ratings vs. number of ratings per movie
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by()
  group_by(movieId) %>%
  summarize(count=n(), avg_rating=mean(rating), genre=first(genres)) %>%
  ggplot(aes(avg_rating, count)) +
  geom_point(aes(size=count)) +
  scale_size_area() +
  ggtitle("Movie Average Rating vs. Number of Ratings ") +
  labs(y="Number of ratings", x="Average rating")

################################
# Explore monthly/seasonal effects
################################
# Monthly boxplot 
edx %>%
  mutate(month=factor(month.abb[month(as.Date(as.POSIXlt(timestamp, origin="1970-01-01")))])) %>%
  ungroup() %>%
  arrange(month) %>%
  ggplot(aes(month,rating)) +
  geom_boxplot(fill="deepskyblue2") +
  labs(x="Month", y="Average rating") +
  ggtitle("Average Rating by Month") +
  scale_x_discrete(limits = month.abb)

################################
# Explore genre effect on rating variability
################################
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n(), avg_ratings=mean(rating)) %>%
  ggplot(aes(reorder(genres, avg_ratings), avg_ratings, fill=avg_ratings)) +
  geom_bar(stat="identity") +
  coord_flip() +
  ggtitle("Average Rating by Movie Genre") +
  labs(x="Genre", y="Average rating")

################################
# Training the model
################################
# Optimize and select the value lambda and train the model
mu_hat <- mean(edx$rating) 
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
    movie_avgs <- edx %>% 
      group_by(movieId) %>% 
      summarize(b_i = sum(rating - mu_hat)/(n()+l), n_i = n())
    
    user_avgs <- edx %>% 
      left_join(movie_avgs, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u=sum(rating-mu_hat-b_i)/(n()+l), n_i = n())
    
    predicted_ratings <- edx %>%
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      mutate(pred = mu_hat + b_i +b_u) %>%
      pull(pred)
    return(RMSE(predicted_ratings, edx$rating))
})
  
qplot(lambdas, rmses)

# Selecting lambda that minimizes the RMSE
lambda <- lambdas[which.min(rmses)]   

################################
# Apply the model on the validation set
################################
mu <- mean(validation$rating) # mean ratings in the validation data
  
# Movie effect
b_i <- validation %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  
# User effect
  b_u <- validation %>%
    left_join(b_i, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))
  
# Generate predictions based on movie and user effects
predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i +  b_u) 
  
# Calculate RMSE for the final model
RMSE(predicted_ratings$pred, validation$rating)
  
################################
# Check the model residuals
################################
# QQ plot of residuals
predicted_ratings %>%
    mutate(residuals=rating-pred) %>%
    ggplot(aes(sample=residuals)) +
    stat_qq(fill="deepskyblue2") + 
    stat_qq_line() +
    ggtitle("QQ-plot of Residuals")
  
# Distribution of residuals
predicted_ratings %>%
    mutate(residuals=rating-pred) %>%
    ggplot(aes(residuals)) +
    geom_histogram(fill="deepskyblue2") +
    ggtitle("Distribution of Model Residuals")
