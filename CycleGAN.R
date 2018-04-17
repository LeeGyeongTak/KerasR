library(keras)
library(progress)
library(abind)

# Functions ---------------------------------------------------------------
img_shape<-c(128,128,3)
main_input<-imgb
build_generator <- function(){
  
  gf<-32
   

  
   
   main_input <- layer_input(shape = img_shape, name = 'main_input')

   d1<-main_input %>% layer_conv_2d(filters=gf,kernel_size= c(4,4),strides = c(2,2), padding = "same") %>%
     layer_activation_leaky_relu(0.2)
   d2<-  layer_conv_2d(d1,gf*2, c(4,4),strides = c(2,2), padding = "same")%>%
     layer_activation_leaky_relu(0.2) %>%
   layer_batch_normalization(momentum = 0.8) 
   
   d3<-layer_conv_2d(d2,gf*4, c(4,4),strides = c(2,2), padding = "same")%>%
     layer_activation_leaky_relu(0.2) %>% 
   layer_batch_normalization(momentum = 0.8)
   
   d4 <- layer_conv_2d(d3,gf*8, c(4,4),strides = c(2,2), padding = "same") %>%
     layer_activation_leaky_relu(0.2) %>%
   layer_batch_normalization(momentum = 0.8)


   u1 <- d4 %>% layer_upsampling_2d(size = c(2, 2)) %>%
     layer_conv_2d(gf*4, c(4,4),strides = c(1,1), padding = "same", activation = "relu")%>%
     layer_batch_normalization(momentum = 0.8) %>% 
     layer_dropout(0.25) 
      u1<-layer_concatenate(c(u1,d3))

    u2 <- u1 %>% layer_upsampling_2d(size = c(2, 2)) %>%
     layer_conv_2d(gf*2, c(4,4),strides = c(1,1), padding = "same", activation = "relu")%>%
     layer_batch_normalization(momentum = 0.8) %>%
      layer_dropout(0.25)
      u2<-layer_concatenate(c(u2,d2))

     u3 <- u2 %>% layer_upsampling_2d(size = c(2, 2)) %>%
     layer_conv_2d(gf, c(4,4),strides = c(1,1), padding = "same", activation = "relu")%>%
     layer_batch_normalization(momentum = 0.8) %>%
       layer_dropout(0.25)
     u3<-layer_concatenate(c(u3,d1))
    u4<-u3 %>% layer_upsampling_2d(size = c(2, 2))

    output_img<-u4 %>% layer_conv_2d(3, c(4,4),strides = c(1,1), padding = "same", activation = "tanh")

   keras_model(main_input, output_img)
   
}

build_discriminator <- function(){
  
  df<-64
  
  
  main_input <- layer_input(shape = img_shape, name = 'main_input')
  
  d1<-main_input %>% layer_conv_2d(df, c(4,4),strides = c(2,2), padding = "same") %>%
    layer_activation_leaky_relu(0.2) 

  
  d2<-  layer_conv_2d(d1,df*2, c(4,4),strides = c(2,2), padding = "same")  %>%
    layer_activation_leaky_relu(0.2) %>% 
    layer_batch_normalization(momentum = 0.8) 
  
  d3<-layer_conv_2d(d2,df*4, c(4,4),strides = c(2,2), padding = "same")  %>%
    layer_activation_leaky_relu(0.2) %>%
  layer_batch_normalization(momentum = 0.8)
  
  d4 <- layer_conv_2d(d3,df*8, c(4,4),strides = c(2,2), padding = "same")  %>%
    layer_activation_leaky_relu(0.2) %>%
  layer_batch_normalization(momentum = 0.8)
  
  validiy<- d4 %>% 
  layer_conv_2d(1, c(4,4),strides = c(1,1), padding = "same")
  
  keras_model(main_input, validiy)
}



# Parameters --------------------------------------------------------------

# Batch and latent size taken from the paper
epochs <- 50

patch<-(128/2**4)
disc_patch<-c(patch,patch,1)
adam_lr <- 0.0002 
adam_beta_1 <- 0.5
img_shape<-c(128,128,3)

# Model Definition --------------------------------------------------------
# Build the generator
generatorab <- build_generator()
generatorab %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = "binary_crossentropy"
)

generatorba <- build_generator()
generatorba %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = "binary_crossentropy"
)


imga <- layer_input(shape =(c(128,128,3)))
imgb <- layer_input(shape = (c(128,128,3)))

fakeb <- generatorab((imga))
fakea <- generatorba((imgb))

recona <- generatorba((fakeb))
reconb <- generatorab((fakea))



# Build the discriminator
discriminatora <- build_discriminator()
discriminatora %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("mean_squared_error"),metrics='accuracy'
)
discriminatorb <- build_discriminator()
discriminatorb %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("mean_squared_error"),metrics='accuracy'
)
freeze_weights(discriminatora)
freeze_weights(discriminatorb)
valida<-discriminatora(fakea)
validb<-discriminatorb(fakeb)



combined <- keras_model(list(imga, imgb), list(valida,validb,fakeb,fakea,recona,reconb))
combined %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),loss_weights=list(1,1,0,0,10,10),
  loss = list("mean_squared_error", "mean_squared_error","mean_absolute_error","mean_absolute_error","mean_absolute_error","mean_absolute_error")
)

# Data Preparation --------------------------------------------------------




# install.packages("imager")
library(imager)
library("EBImage")
j<-1
dataset<-"apple2orange"
dataset<-"horse2zebra"
## you should download image data 
# Training ----------------------------------------------------------------
setwd(paste0("E:\\gyeongtaek\\kerasr\\",dataset,"\\",dataset,"\\trainA"))
traina<-list()
file_list<-list.files()
for(j in 1:length(file_list)){
da<-load.image(file_list[j])
da2<-da[,,1,]
da3<-resize(da2,128,128)
traina[[j]]<-da3
}

require(abind)
trainA <- abind(traina,along=0)
dim(trainA) <-c(dim(trainA)[1],128,128,3)

setwd(paste0("E:\\gyeongtaek\\kerasr\\",dataset,"\\",dataset,"\\trainB"))
trainb<-list()
file_list<-list.files()
K<-1
for(j in 1:length(file_list)){
  da<-load.image(file_list[j])
  da2<-da[,,1,]
  if(is.na(dim(da2)[3])){
    next;
  }
  
  da3<-resize(da2,128,128)

  trainb[[j]]<-da3
  K<-j+1
}

require(abind)
trainB <- abind(trainb[-309],along=0)
dim(trainB) <-c(dim(trainB)[1],128,128,3)
dim(trainB)

trainA<-(trainA - 0.5)/0.5
trainB<-(trainB - 0.5)/0.5


# TEST data ----------------------------------------------------------------
setwd(paste0("E:\\gyeongtaek\\kerasr\\",dataset,"\\",dataset,"\\testA"))
testa<-list()
file_list<-list.files()
for(j in 1:length(file_list)){
  da<-load.image(file_list[j])
  da2<-da[,,1,]
  da3<-resize(da2,128,128)
  testa[[j]]<-da3
}

require(abind)
testA <- abind(testa,along=0)
dim(testA) <-c(dim(testA)[1],128,128,3)

setwd(paste0("E:\\gyeongtaek\\kerasr\\",dataset,"\\",dataset,"\\testB"))
testb<-list()
file_list<-list.files()
for(j in 1:length(file_list)){
  da<-load.image(file_list[j])
  da2<-da[,,1,]
  da3<-resize(da2,128,128)
  
  testb[[j]]<-da3
}
testB <- abind(testb,along=0)
dim(testB) <-c(dim(testB)[1],128,128,3)



testA<-(testA - 0.5)/0.5
testB<-(testB - 0.5)/0.5

daloss <- NULL
dbloss <- NULL
dloss<-NULL
g_loss<-NULL
for(epoch in 1:200){
  
  num_batches <- 100
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
 
  

  batch_size <-20
  for(index in 1:num_batches){
    
    pb$tick()
    
    sama<-sample(1:dim(trainA)[1],batch_size)
    samb<-sample(1:dim(trainB)[1],batch_size)
    imgsa<-trainA[sama,,,]
    imgsb<-trainB[samb,,,]
    dim(imgsa)
    fakeb <- predict(generatorab,imgsa)
    fakea <- predict(generatorba,imgsb)
    
    valid<-array(1,dim=c(disc_patch,batch_size))  
    fake<-array(0,dim=c(disc_patch,batch_size))
    
    ak<-predict(discriminatora,imgsa)

    dim(valid)<-dim(ak)
    dim(fake)<-dim(ak)
   
    # dim(imga)
    da_loss_real <- train_on_batch(
      discriminatora, x = imgsa, 
      y =valid
    )
    da_loss_fake <- train_on_batch(
      discriminatora, x = fakea, 
      y =fake
    )

    daloss<-c(daloss,0.5*(mean(unlist(da_loss_fake))+
    mean(unlist(da_loss_real))) )
      
      
    db_loss_real <- train_on_batch(
      discriminatorb, x = imgsb, 
      y =valid
    )
    db_loss_fake <- train_on_batch(
      discriminatorb, x = fakeb, 
      y =fake
    )
    
    dbloss<-c(dbloss,0.5*(mean(unlist(db_loss_fake))+
                            mean(unlist(db_loss_real))) )
      
      
    dloss<-c(dloss,0.5*(0.5*(mean(unlist(da_loss_fake))+
                               mean(unlist(da_loss_real))) +0.5*(mean(unlist(db_loss_fake))+
                                                                   mean(unlist(db_loss_real)))))
    
    sama<-sample(1:dim(trainA)[1],batch_size)
    samb<-sample(1:dim(trainB)[1],batch_size)
    imgsa<-trainA[sama,,,]
    imgsb<-trainB[samb,,,]

    g_loss0 <- train_on_batch(
      combined, 
      list(imgsa, imgsb),
      list(valid, valid,imgsb,imgsa,imgsa,imgsb)
    )
    
    g_loss<-rbind(g_loss,unlist(g_loss0))
    
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  
  sama<-sample(1:dim(trainA)[1],5)
  samb<-sample(1:dim(trainB)[1],5)
  testimga<-trainA[sama,,,]
  testimgb<-trainB[samb,,,]
  

  
  # Get a batch to display
  testfakeb <- predict(
    generatorab,    
    testimga
  )
  testfakea <- predict(
    generatorba,    
    testimgb
  )
  testreconsta <- predict(
    generatorba,    
    testfakeb
  )
  testreconstb <- predict(
    generatorab,    
    testfakea
  )
  
  
  dim(testfakea)<-c(5,128,128,3)
  dim(testfakeb)<-c(5,128,128,3)
  dim(testreconsta)<-c(5,128,128,3)
  dim(testreconstb)<-c(5,128,128,3)
  k<-4
  for(k in 1:5){
  oria<-testimga[k,,,]
  orib<-testimgb[k,,,]
  dim(oria)<-c(128,128,3)
  dim(orib)<-c(128,128,3)
    
  dg<-testfakea[k,,,]
  dg2<-testfakeb[k,,,]
  dg3<-testreconsta[k,,,]
  dg4<-testreconstb[k,,,]
  
  min(oria)
  max(oria)
  min(dg)
  max(dg)
  min((dg+1)/2)
  max((dg+1)/2)
  par(mfrow=c(3,2)) 
dim(dg)
round((oria+1)/2,4) %>% as.raster() %>%
  plot()
round(((orib+1)/2),4) %>% as.raster() %>%
  plot()
  round(((dg) + 1.00)/2,4) %>% as.raster() %>%
    plot()
  round(((dg2) + 1.00)/2,4) %>% as.raster() %>%
    plot()
  round(((dg3) + 1.00)/2,4) %>% as.raster() %>%
    plot()
  round(((dg4) + 1.00)/2,4) %>% as.raster() %>%
    plot()
  }
  
}
