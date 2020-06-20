library(RTextTools)
library(tm)
library(NLP)
library(caret)
library(wordcloud)
library(xlsx)


#Input raw data given by professor as it is
RawData = read.csv("C:/Users/Buddabathuni/Desktop/InputAnalysisFile.csv")
View(RawData)

#cleaning the data, removing numbers, punctuation, stopwords
x=RawData$Request
x=as.character(x)
x=tolower(x)
x=removePunctuation(x)
x=removeNumbers(x)
stopWords = stopwords("en")
extrastop = c("au", "url", "ive", "im", "ill","a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "article", "page", "just")
x=removeWords(x,stopWords)
x=removeWords(x,extrastop)
tokenize_ngrams <- function(x, n=3)
return(rownames(as.data.frame(unclass(textcnt(x,method="string",n=n)))))


myTdm1 <- as.DocumentTermMatrix(slam::as.simple_triplet_matrix(matrix(c(rep(1, 101), rep(1,1), rep(0, 100)), ncol=2)), weighting = weightTf)
removeSparseTerms(myTdm1, .99)


RawData$Request=x

View(RawData)
wordcloud(RawData$Request, max.words = 100, random.order = FALSE)

#Convert politeness column to numeric
y = RawData$politeness
y=as.character(y)
y[y == "polite"] <- "1"
y[y == "impolite"] <- "-1"
y[y == "neutral"] <- "0"
RawData$politeness = y


#matrix
dtMatrix <- create_matrix(RawData["Request"],language="english", removeStopwords=TRUE)

# Configure the training data
container <- create_container(dtMatrix, RawData$politeness, trainSize=1:3400, virgin=FALSE)

# train a SVM Model
model <- train_model(container, "SVM", kernel="linear", cost=1)

#testing data
testData = as.data.frame(RawData[3401:4000,2], drop=FALSE)
View(testData)
trace("create_matrix", edit=T)

# create a prediction document term matrix
predMatrix <- create_matrix(testData, originalMatrix=dtMatrix)
View(testData)

# create the corresponding container
predictionContainer <- create_container(predMatrix, labels=rep(0,600), testSize=1:600, virgin=FALSE)
results <- classify_model(predictionContainer, model)
View(results)

confusionMatrix(check$politeness,results$SVM_LABEL)


#converting back to the original label
z = results$SVM_LABEL
z=as.character(z)
z[z == "1"] <- "polite"
z[z == "-1"] <- "impolite"
z[z == "0"] <- "neutral"
results$SVM_LABEL = z

FinalFile = data.frame(testData$`RawData[3401:4000, 2]`, results$SVM_LABEL,results$SVM_PROB)

names(FinalFile)[1]<-paste("Query")
names(FinalFile)[2]<-paste("Politeness_result")
names(FinalFile)[3]<-paste("Probability")

View(FinalFile)

write.xlsx(FinalFile, "C:/Users/Buddabathuni/Desktop/Results.xlsx")


