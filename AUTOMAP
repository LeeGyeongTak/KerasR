# Image reconstruction by domain-transform manifold learning (AUTOMAP)
# source("http://bioconductor.org/biocLite.R")
# biocLite("EBImage")
library(R.matlab)
library(jpeg)
library("EBImage")

setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_kspace_Under")
file<-list.files()
nume_index<-as.numeric(gsub("im|.mat","",file))
file<-file[order(nume_index)]
train_list<-list()
for(j in 1:length(file)){
  setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_kspace_Under")
  data<-readMat(file[j])
  realdata<-(data$f.im.u)## K-space Data
  train_list[[j]]<-as.vector(realdata[seq(1,256,by=3),seq(1,256,by=3)])
  setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_kspace_Under_image")
  cat("\n",j)
}
setwd("E:\\gyeongtaek\\의료인공지능")
train_list[[1]]
save(train_list,file="train_sensor.RData")



setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_image_label")
file<-list.files()
nume_index<-as.numeric(gsub("im|.mat","",file))
file<-file[order(nume_index)]
train_label<-list()
for(j in 1:length(file)){
  setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_image_label")
  data<-readMat(file[j])

  realdata<-(data$im.org)
  da3<-resize(realdata,86,86)
 
  train_label[[j]]<-as.vector(da3)
  setwd("E:\\gyeongtaek\\의료인공지능\\MRI_KSPACE\\Train_image_label_image") ##full reconst image
  writeJPEG((da3),paste0("im",j,".jpg"))
  cat("\n",j)
}
setwd("E:\\gyeongtaek\\의료인공지능")
dim(train_list[[1]])

save(train_list,file="train_sensor.RData")
save(train_label,file="train_label.RData")


setwd("E:\\gyeongtaek\\의료인공지능")
load("train_label.RData")
load("train_sensor.RData")
library(abind)
trainA <- abind(train_list,along=0)
trainB <- abind(train_label,along=0)
img_shape<-c(86,86,1)
dim(trainB)<-c(894,img_shape)

library(progress)
library(keras)
noise_shape<-c(100)
input_size<-c(86*86)

build_AUTOMAP<- function(){
  
  cnn <- keras_model_sequential()
  
  cnn %>% layer_dense(input_size,input_shape = c(input_size), activation = "tanh") %>%
  layer_dense(input_size,activation = "tanh")    %>%
  layer_reshape(c(86,86,1),input_shape = input_size) %>%


  layer_conv_2d(128, c(5,5), padding = "same", strides = c(1,1)) %>%
  layer_activation_leaky_relu(0.3) %>%
  layer_dropout(0.25) %>%  
  layer_batch_normalization(momentum = 0.8)%>%

  layer_conv_2d(128, c(5, 5), padding = "same", strides = c(1,1)) %>%
  layer_activation_leaky_relu(0.3) %>%
  layer_dropout(0.25) %>%  
  layer_batch_normalization(momentum = 0.8)%>%
  layer_conv_2d(128, c(7, 7), padding = "same") %>%
  layer_dropout(0.25) %>%  
  layer_batch_normalization(momentum = 0.8)%>%
  layer_conv_2d(1, c(3, 3), padding = "same")
  
  sensor <- layer_input(shape = input_size)
  image <- cnn(sensor)
   
  
  keras_model(sensor, image)
}





opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)
AUTOMAP<-build_AUTOMAP()
AUTOMAP %>% compile(
  optimizer = opt,
  loss = "mean_squared_error"
)


sam<-sample(1:nrow(trainA),850) ## train 850/ test 44
batch_size <- 10

tb<-trainB[sam,,,]
dim(tb)<-c(850,86,86,1)
dim(trainB) 

AUTOMAP %>% fit(
  trainA[sam,], tb,
  batch_size = batch_size,
  epochs =800,
  shuffle = TRUE
)

dim(trainA)
dim(trainB)
image_batch <- trainA[-sam,] ## test
image_batch <- trainA[1:5,] ## train
dim(image_batch)
generated_images <- predict(AUTOMAP,(image_batch[,1:(86*86)]))



img <- NULL
for(i in 1:5){
  img <- cbind(img, generated_images[i,,,])
}

round((img)/2+0.08,5) %>% as.raster() %>%
  plot()



tr_i2<-trainB[1:5,,,]  ## train True
tr_i2<-trainB[-sam,,,] ## test  True


dim(tr_i2)<-c(5,86,86,1)
dim(tr_i2)<-c(44,86,86,1)

img <- NULL
for(i in 1:5){
  img <- cbind(img, tr_i2[i,,,])
}
min((img ))
max(img)
round((img)/2+0.1,5) %>% as.raster() %>%
  plot()
