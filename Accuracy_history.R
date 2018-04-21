batch_size <- 32
# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

    x_train <- cifar10$train$x/255
    x_test <- cifar10$test$x/255
    bf_y<-cifar10$train$y    
    y_train <- to_categorical(cifar10$train$y, num_classes = 10)
    y_test <- to_categorical(cifar10$test$y, num_classes = 10)
    bf_y2<-cifar10$test$y
    
    # Defining Model ----------------------------------------------------------
    
    # Initialize sequential model
    model <- keras_model_sequential()
    
    model %>%
      
      # Start with hidden 2D convolutional layer being fed 32x32 pixel images
      layer_conv_2d(
        filter = 32, kernel_size = c(3,3), padding = "same", 
        input_shape = c(32, 32, 3)
      ) %>%
      layer_activation("relu") %>%
      
      # Second hidden layer
      layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
      layer_activation("relu") %>%
      
      # Use max pooling
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_dropout(0.25) %>%
      
      # 2 additional hidden 2D convolutional layers
      layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
      layer_activation("relu") %>%
      layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
      layer_activation("relu") %>%
      
      # Use max pooling once more
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_dropout(0.25) %>%
      
      # Flatten max filtered output into feature vector 
      # and feed into dense layer
      layer_flatten() %>%
      layer_dense(512) %>%
      layer_activation("relu") %>%
      layer_dropout(0.5) %>%
      
      # Outputs from dense layer are projected onto 10 unit output layer
      layer_dense(10) %>%
      layer_activation("softmax")
    
    opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)
    # model<-multi_gpu_model(model,gpus=2)
    model %>% compile(
      loss = "categorical_crossentropy",
      optimizer = opt,
      metrics = "accuracy"
    )
     
  

 #####################
  AccuracynHistory <- R6::R6Class("AccuracynHistory",
                                    inherit = KerasCallback, 
                                    
                                    public = list(
                                      val = NULL,tval=NULL,
                                      on_epoch_end = function(epoch,logs = list()) {
                                        preds <- do.call(cbind,list(model$predict(x_test)))
                                        value<-sum(apply(preds,1,which.max)-1==bf_y2)/length(bf_y2)
                                        self$val <- c(self$val,value)
                                        
                                        tpreds <- do.call(cbind,list(model$predict(x_train)))
                                        tvalue<-sum(apply(tpreds,1,which.max)-1==bf_y)/length(bf_y)
                                        self$tval <- c(self$tval,tvalue)
                                        

                                      }
                                    )
    )
     predictions <- AccuracynHistory$new()

   modd<- model %>% fit(
      x_train, y_train,
      batch_size = batch_size,
      epochs = 50,
     
      shuffle = TRUE,callbacks=predictions
    )  
   
   
   ts.plot(cbind(predictions$val,predictions$tval),col=c("red","blue"))
