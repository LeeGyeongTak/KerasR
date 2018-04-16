# install.packages("keras")
library(progress)
library(keras)
# install_keras()



residual_blcok<-function(layerinput){
  d<-layerinput %>% 
    layer_conv_2d(filters=64,kernel_size= c(3,3),strides = c(1,1), padding = "same", activation ="relu") %>%
    layer_batch_normalization(momentum = 0.8) %>%
    layer_conv_2d(filters=64,kernel_size= c(3,3),strides = c(1,1), padding = "same", activation ="relu")  %>%
    layer_batch_normalization(momentum = 0.8)
  layer_add(c(d,layerinput))
}

deconv2d<-function(layerinput){
  u<-layerinput %>% layer_upsampling_2d(size = c(2, 2)) %>%
    layer_conv_2d(filters=256,kernel_size= c(3,3),strides = c(1,1), padding = "same", activation ="relu")  
  u
}

layerinput<-d0
filters<-64
d_block<-function(layerinput,filters,strides=1,bn=TRUE){
  d<-layerinput %>% 
    layer_conv_2d(filters,kernel_size= c(3,3),strides = strides, padding = "same") %>%
    layer_activation_leaky_relu(0.2) 
  if(bn){
    d<-d %>% layer_batch_normalization(momentum = 0.8) 
  }
  d
}

# Build the generator
build_generator <- function(){
  
  
  main_input <- layer_input(shape = img_shape, name = 'main_input')
  
  d1<-main_input %>% layer_conv_2d(filters=64,kernel_size= c(9,9),strides = c(1,1), padding = "same", activation="relu")
  r<-residual_blcok(d1)
  for(p in 1:16){
    r<-residual_blcok(r)
  }
  
  c2<-r %>% layer_conv_2d(filters=64,kernel_size= c(3,3),strides = c(1,1), padding = "same") %>%
    layer_batch_normalization(momentum = 0.8)
  c2<-layer_add(c(c2,d1))
  # 
  u1<-deconv2d(c2)
  u2<-u1 %>% layer_upsampling_2d(size = c(2, 2)) 
  
  output_img<-u2 %>% layer_conv_2d(3, c(9,9),strides = c(1,1), padding = "same", activation = "tanh")
  
  keras_model(main_input, output_img)
  
}

build_discriminator<-function(){
  df<-64
  d0 <- layer_input(shape = sr_shape,name="d0")
  d1<-d_block(d0,df,bn=F) 
  d2<-d_block(d1,df,strides = 2) 
  d3<-d_block(d2,df*2) 
  d4<-d_block(d3,df*2,strides = 2) 
  d5<-d_block(d4,df*4,bn=F) 
  d6<-d_block(d5,df*4,strides = 2) 
  d7<-d_block(d6,df*8,bn=F)
  d8<-d_block(d7,df*8,strides = 2)
  
  d9<-d8 %>% layer_dense(df*16) %>% layer_activation_leaky_relu(0.2)
  validty<-   d9 %>% layer_dense(1,activation="sigmoid")
  
  keras_model(d0, validty)
}


build_vgg<-function(){
  vggnet <- application_vgg16(weights = 'imagenet', include_top = FALSE)
  vggnet$output_layers<-vggnet$layers[[9]]$output
  vggsr <- layer_input(shape =(sr_shape))
  fake_features<-vggnet(vggsr)
  
  keras_model(vggsr,fake_features)
}


# Parameters --------------------------------------------------------------



img_shape<-c(64,64,3)
sr_shape<-img_shape*4
sr_shape[3]<-3
# layerinput<-main_input


batch_size <- 100

patch<-(sr_shape[1]/2**4)
disc_patch<-c(patch,patch,1)


adam_lr <- 0.00005 
adam_beta_1 <- 0.5
# img_shape<-c(128,128,3)
# img_shape<-c(224,224,3)

# Model Definition --------------------------------------------------------
vggnet<-build_vgg()
freeze_weights(vggnet)
vggnet %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("mean_squared_error"),metrics='accuracy'
)

# Build the generator
generator <- build_generator()
generator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = "binary_crossentropy"
)


# Build the discriminator
discriminator <- build_discriminator()
discriminator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("mean_squared_error"),metrics='accuracy'
)

imgir <- layer_input(shape =(img_shape))
imgsr <- layer_input(shape = (sr_shape))


fake_sr <- generator((imgir))


fake_feature<-vggnet(fake_sr)
freeze_weights(discriminator)

validty<-discriminator(fake_sr)

combined <- keras_model(list(imgir, imgsr), list(validty,fake_feature))
combined %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy", "mean_squared_error"),loss_weights = list(0.001,1)
)


combined <- keras_model((imgir), (validty))
combined %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy"),metrics='accuracy'
)

combined2 <- keras_model(list(imgsr), list(fake_feature))
combined2 %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("mean_squared_error"),loss_weights = list(1)
)
# install.packages("EBImage")
library(imager)
library("EBImage")
j<-1
dataset<-"apple2orange"
dataset<-"horse2zebra"
# Training ----------------------------------------------------------------
setwd(paste0("E:\\gyeongtaek\\kerasr\\",dataset,"\\",dataset,"\\trainA"))
traina<-list()
trainsr<-list()
file_list<-list.files()
for(j in 1:length(file_list)){
  da<-load.image(file_list[j])
  da2<-da[,,1,]
  da3<-resize(da2,64,64)
  trainsr[[j]]<-da2
  traina[[j]]<-da3
}

require(abind)
train <- abind(traina,along=0)
dim(train) <-c(dim(train)[1],64,64,3)

trainSR <- abind(trainsr,along=0)
dim(trainSR) <-c(dim(trainSR)[1],256,256,3)


train<-(train - 0.5)/0.5
trainSR<-(trainSR - 0.5)/0.5


dloss<-NULL
g_loss<-NULL

epochs <- 50
for(epoch in 1:epochs){
  
  num_batches <- 100
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
  batch_size <-10
  
  for(index in 1:num_batches){
    
    pb$tick()
    
    sama<-sample(1:dim(train)[1],batch_size)
    imga<-train[sama,,,]
    imgSR<-trainSR[sama,,,]
    
    
    fake_sr <- predict(generator,imga)
 
    valid<-array(1,dim=c(disc_patch,batch_size))  
    fake<-array(0,dim=c(disc_patch,batch_size))
  
    dim(valid)<-c(batch_size,16,16,1)
    dim(fake)<-c(batch_size,16,16,1)
    
    
    d_loss_real <- train_on_batch(
      discriminator, x = imgSR, 
      y =valid
    )
    d_loss_fake <- train_on_batch(
      discriminator, x = fake_sr, 
      y =fake
    )
    
    dloss<-c(dloss,0.5*(mean(unlist(d_loss_fake))+
                           mean(unlist(d_loss_real))) )
    
    
    
    sama<-sample(1:dim(train)[1],batch_size)
    imga<-train[sama,,,]
    imgSR<-trainSR[sama,,,]
    image_feature<-predict(vggnet,imgSR)
    image_feateres<-vggnet$predict(imgSR)
 
    g_loss0 <- train_on_batch(
      combined, 
      list(imga, imgSR),
      list(valid, image_feateres)
    )

    g_loss<-rbind(g_loss,unlist(g_loss0))
    
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  
  sama<-sample(1:dim(train)[1],5)

  imga<-train[sama,,,]
  imgsr<-trainSR[sama,,,]
  
  fake_sr <- predict(generator,imga)
  
  imga <- 0.5*imga + 0.5
  fake_sr <- 0.5*fake_sr+0.5
  imgsr <- 0.5*imgsr+0.5
  
  par(mfrow=c(1,3)) 
  
  for(k in 1:5){
    
    dg<-imga[k,,,]
    dg2<-fake_sr[k,,,]
    dg3<-imgsr[k,,,]
    
    
    round(((dg)),4) %>% as.raster() %>%
      plot()
    round(((dg2)),4) %>% as.raster() %>%
      plot()
    round(((dg3)),4) %>% as.raster() %>%
      plot()
    
  }
  
}



