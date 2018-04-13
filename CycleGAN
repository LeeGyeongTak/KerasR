library(keras)
library(progress)
library(abind)
# k_set_image_data_format('channels_first')

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
adam_lr <- 0.00005 
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
# Training ----------------------------------------------------------------
setwd("E:\\gyeongtaek\\kerasr\\\edges2shoes\\train")
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

setwd("E:\\gyeongtaek\\kerasr\\apple2orange\\apple2orange\\trainb")
trainb<-list()
file_list<-list.files()
for(j in 1:length(file_list)){
  da<-load.image(file_list[j])
  da2<-da[,,1,]
  da3<-resize(da2,128,128)

  trainb[[j]]<-da3
}

require(abind)
trainB <- abind(trainb,along=0)
dim(trainB) <-c(dim(trainB)[1],128,128,3)
dim(trainB)

trainA<-(trainA - 0.5)/0.5
trainB<-(trainB - 0.5)/0.5

dim(da3)
dim(da3)
head(da)
min(trainA[1,,,])
max(trainA[1,,,])

dim(da2)
# setwd("E:\\gyeongtaek\\kerasr\\apple2orange\\apple2orange\\trainB")

daloss <- NULL
dbloss <- NULL
dloss<-NULL
g_loss<-NULL
for(epoch in 1:epochs){
  
  num_batches <- 100
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
 
  

  batch_size <-32
  for(index in 1:num_batches){
    
    pb$tick()
    
    sama<-sample(1:dim(trainA)[1],batch_size)
    samb<-sample(1:dim(trainB)[1],batch_size)
    imga<-trainA[sama,,,]
    imgb<-trainB[samb,,,]
    dim(imga)
    fake_B <- predict(generatorab,imga)
    fake_A <- predict(generatorba,imgb)
    
    valid<-array(1,dim=c(disc_patch,32))  
    fake<-array(0,dim=c(disc_patch,32))
    
    ak<-predict(discriminatora,imga)

    dim(valid)<-dim(ak)
    dim(fake)<-dim(ak)
   
    # dim(imga)
    da_loss_real <- train_on_batch(
      discriminatora, x = imga, 
      y =valid
    )
    da_loss_fake <- train_on_batch(
      discriminatora, x = fake_A, 
      y =fake
    )

    daloss<-c(daloss,0.5*(mean(unlist(da_loss_fake))+
    mean(unlist(da_loss_real))) )
      
      
    db_loss_real <- train_on_batch(
      discriminatorb, x = imgb, 
      y =valid
    )
    db_loss_fake <- train_on_batch(
      discriminatorb, x = fake_B, 
      y =fake
    )
    
    dbloss<-c(dbloss,0.5*(mean(unlist(db_loss_fake))+
                            mean(unlist(db_loss_real))) )
      
      
    dloss<-c(dloss,0.5*(daloss+dbloss))
    
    # sama<-sample(1:dim(trainA)[1],batch_size)
    # samb<-sample(1:dim(trainB)[1],batch_size)
    imga<-trainA[sama,,,]
    imgb<-trainB[samb,,,]
    
    
    
    # valid<-array(1,dim=c(disc_patch,32))  
    # fake<-array(0,dim=c(disc_patch,32))
      
  
    # (valida,validb,fakeb,fakea,recona,reconb))
    g_loss0 <- train_on_batch(
      combined, 
      list(imga, imgb),
      list(valid, valid,imga,imgb,imga,imgb)
    )
    
    g_loss<-rbind(g_loss,unlist(g_loss0))
    
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  # Evaluate the testing loss here
  sama<-sample(1:dim(trainA)[1],5)
  samb<-sample(1:dim(trainB)[1],5)
  imga<-trainA[sama,,,]
  imgb<-trainB[samb,,,]
  
  
  # Get a batch to display
  fakeb <- predict(
    generatorab,    
    imga
  )
  fakea <- predict(
    generatorba,    
    imgb
  )
  reconsta <- predict(
    generatorba,    
    fakeb
  )
  reconstb <- predict(
    generatorab,    
    fakea
  )
  
  
  dim(fakea)<-c(5,128,128,3)
  dim(fakeb)<-c(5,128,128,3)
  dim(reconsta)<-c(5,128,128,3)
  dim(reconstb)<-c(5,128,128,3)
  k<-2
  for(k in 1:5){
  oria<-imga[k,,,]
  orib<-imgb[k,,,]
  dim(oria)<-c(128,128,3)
  dim(orib)<-c(128,128,3)
    
  dg<-fakea[k,,,]
  dg2<-fakeb[k,,,]
  dg3<-reconsta[k,,,]
  dg4<-reconstb[k,,,]
  
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


