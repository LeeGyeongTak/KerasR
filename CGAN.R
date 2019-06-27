# install.packages("progress")
library(progress)
library(keras)
noise_shape<-c(100)

img_shape<-c(28,28,1)
build_discriminator <- function(){
  
  
  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_conv_2d(
      64, c(3,3), padding = "same", strides = c(2,2),
      input_shape = img_shape
    ) %>%
    layer_activation_parametric_relu() %>%
    layer_dropout(0.25) %>%
    
    layer_conv_2d(128, c(3, 3), padding = "same", strides = c(2,2)) %>%
    layer_zero_padding_2d(padding=list(c(0,1),c(0,1))) %>%
    layer_activation_parametric_relu() %>%
    layer_dropout(0.25) %>%  
    layer_batch_normalization(momentum = 0.8) %>%
    layer_conv_2d(256, c(3, 3), padding = "same", strides = c(2,2)) %>%
    layer_activation_parametric_relu() %>%
    layer_dropout(0.25) %>%  
    layer_batch_normalization(momentum = 0.8) %>%
    layer_conv_2d(512, c(3, 3), padding = "same", strides = c(2,2)) %>%
    layer_activation_parametric_relu() %>%
    layer_dropout(0.25) %>%  
    layer_flatten() 
  
  
  image <- layer_input(shape = img_shape)
  features <- cnn(image)
  validity <- features %>%  layer_dense(unit=1,activation="sigmoid" )
  qnet<-features %>%  layer_dense(unit=128,activation="relu" )
  label <- qnet %>% layer_dense(unit=10, activation='softmax')
  
  
  keras_model((image),list(validity,label))

}



build_generator <- function(){
  
  cnn <- keras_model_sequential()
  
  cnn %>%
    layer_dense(128*7*7, activation = "relu",input_shape = noise_shape+10) %>%
    layer_reshape(c(7, 7,128)) %>%
    layer_batch_normalization(momentum = 0.8) %>%
    layer_upsampling_2d()  %>%
    layer_conv_2d(128, c(3, 3), padding = "same") %>%
    layer_activation_parametric_relu%>%
    layer_batch_normalization(momentum = 0.8) %>%
    layer_upsampling_2d()  %>%
    layer_conv_2d(64, c(3, 3), padding = "same") %>%
    layer_activation_parametric_relu %>% 
    layer_batch_normalization(momentum = 0.8) %>%
    layer_conv_2d(1, c(3, 3), padding = "same") %>%
    layer_activation("tanh")
  
  noise <- layer_input(shape = noise_shape+10)
  img <- cnn(noise)
  keras_model(noise, img)
}



epochs <- 50
batch_size <- 100
latent_size <- 100

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr <- 0.00005 
adam_beta_1 <- 0.5
c_given_x<-1


# Model Definition --------------------------------------------------------

# Build the discriminator
discriminator <- build_discriminator()
discriminator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy","sparse_categorical_crossentropy"),metrics='accuracy'
)

# Build the generator
generator <- build_generator()
generator %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = "binary_crossentropy"
)



noise <- layer_input(shape = list(noise_shape+10))

fake <- generator(noise)

freeze_weights(discriminator)
results <- discriminator(fake)
valid<-results[[1]]
label<-results[[2]]

combined <- keras_model(noise, list(valid,label))
combined %>% compile(
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
  loss = list("binary_crossentropy","sparse_categorical_crossentropy")
)




mnist <- dataset_mnist()
mnist$train$x <- (mnist$train$x - 127.5)/127.5
mnist$test$x <- (mnist$test$x - 127.5)/127.5
mnist$train$x <- array_reshape(mnist$train$x, c(60000,  28, 28,1))
mnist$test$x <- array_reshape(mnist$test$x, c(10000, 28, 28,1))

num_train <- dim(mnist$train$x)[1]
num_test <- dim(mnist$test$x)[1]
discloss<-NULL
discloss2<-NULL
generloss<-NULL
# Training ----------------------------------------------------------------
epoch<-1
for(epoch in 1:epochs){
  
  num_batches <- trunc(num_train/batch_size)
  pb <- progress_bar$new(
    total = num_batches, 
    format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
    clear = FALSE
  )
  
  epoch_gen_loss <- NULL
  epoch_disc_loss <- NULL
  
  possible_indexes <- 1:num_train
  index<-1
  num_batches<-num_batches
  sambat<-sample(num_batches)
  for(index in sambat){
    
    pb$tick()
    
    # Generate a new batch of noise
    noise <- runif(n = noise_shape * batch_size, min = -1, max = 1)  %>% matrix(nrow=batch_size)
    
    # Get a batch of real images
    batch <- sample(possible_indexes, size = batch_size)
    possible_indexes <- possible_indexes[!possible_indexes %in% batch]
    image_batch <- mnist$train$x[batch,,,,drop = FALSE]
    label_batch <- mnist$train$y[batch]
    
    
    sampled_labels <- sample(0:9, batch_size, replace = TRUE) %>%
      to_categorical()
    geninput<-cbind(noise,sampled_labels)
    
    generated_images <- predict(generator,geninput)
    
    # Check if the discriminator can figure itself out
    disc_loss_real <- train_on_batch(
      discriminator, x = image_batch, 
      y = list(rep(1,batch_size) %>% matrix(ncol=1),label_batch)
    )
    
    disc_loss_fake <- train_on_batch(
      discriminator, x = generated_images, 
      y = list(rep(0,batch_size) %>% matrix(ncol=1),label_batch)
    )
    
    discloss<-rbind(discloss,unlist(disc_loss_real))
    discloss2<-rbind(discloss2,unlist(disc_loss_fake))
    
    
    
    
    combined_loss <- train_on_batch(
      combined, x=geninput,
      y = list(rep(1,batch_size) %>% matrix(ncol=1),sampled_labels)
    )
      
    
    
    noise <- runif(n = noise_shape * batch_size, min = -1, max = 1)  %>% matrix(nrow=batch_size)
    generloss <- rbind(generloss, unlist(combined_loss))
    
  }
  
  cat(sprintf("\nTesting for epoch %02d:", epoch))
  
  
  cat("\n")
  
  
  noise <- runif(n = noise_shape * 10, min = -1, max = 1)  %>% matrix(nrow=10)
  cate<-0:9 %>% to_categorical()
  geninput<-cbind(noise,cate)
  
  generated_images <- predict(generator,geninput)
  
  img <- NULL
  for(i in 1:10){
    img <- cbind(img, generated_images[i,,,])
  }
  min((img + 1))
  round((img + 1)/2,5) %>% as.raster() %>%
    plot()
  
}

