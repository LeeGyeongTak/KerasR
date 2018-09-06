library(keras)
build_actor<-function(){

  action_picked <- layer_input(shape = c(4))
  input<-layer_input(shape = c(20)) 
  act_prob<- input %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(20)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 4, activation = 'softmax') 
  selected_act_prob <-layer_multiply(c(act_prob,action_picked))
  selected_act_prob<- selected_act_prob %>% layer_lambda( function(x){ k_sum(x,axis=-1,keepdims=T)})
  keras_model(list(input,action_picked), list(act_prob,selected_act_prob))
  
  
}


categorical_crossentropy<-function(y_true,y_pred){
  y_pred<-k_sum(y_pred,0.00001)
  loss<- -y_true * k_log(y_pred)
  loss
  
}

model<-build_actor()
adam_lr <- 0.00005 
adam_beta_1 <- 0.5
model %>% compile(
  loss = list('mean_squared_error',categorical_crossentropy),
  loss_weights = list(0.0,1.0),
  optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1),
)

value_network <- keras_model_sequential()
value_network %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(20)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'linear')

value_network %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_rmsprop()
)





move<-function(x,action){
  
  if(action == "left"){
    if(x[2]-1<1){
      x
    }else{
      x[2]<-x[2]-1
      x
    }
  }
  if(action == "right"){
    if(x[2]+1>ncol(stm)){
      x
    }else{
      x[2]<-x[2]+1
      x
    }
  }
  if(action == "up"){
    if(x[1]-1<1){
      x
    }else{
      x[1]<-x[1]-1
      x
    }
  }
  if(action == "down"){
    if(x[1]+1>nrow(stm)){
      x
    }else{
      x[1]<-x[1]+1
      x
    }
  }
  x
}
next_where<-function(index){ 
  zero<-rep(0,100)
  zero[index]<-1
  zero
  
  
}




#######state matrix 
stm<-matrix(1:100,ncol=10,nrow=10,byrow=T)
wv<-rep(0,nrow(stm))
yv<-rep(0,ncol(stm))
convert_coord<-function(x){
  wv2<-wv
  yv2<-yv
  wv2[x[1]]<-1
  yv2[x[2]]<-1
  c(wv2,yv2)
}

return_reward<-function(state,current_state){
  re_index<-which(state==1)
  
  if(  re_index==100){
    reward<- 100# episode end
    done<-T
  }
  else if(re_index %in% c(31,32,33,34,35,38,42,44,45,50,61,66,67,68,69,70,72)){
    # else if(re_index==12){
    reward<- -1
    done<-F
  }else{
    reward <- -0.1
    done<-F
  }
  if(re_index==which(current_state==1)){
    reward<-reward*2
  }
  # xx<-ceiling(re_index/ 10) ## row
  # yy<-re_index %% 10  ## col
  # yy<-ifelse(yy ==0,10,yy)
  # reward_weight<-sqrt((yy-10)^2+(xx-10)^2) #weigthed reward by distance from current state to goal
  # reward<-reward*reward_weight/sqrt(200)
  c(reward,done)
  
}

action<-c("left","right","down","up")

state_size <-ncol(stm)*nrow(stm)


epoch<-50
mini_batch<-20
init_data<-convert_coord(c(1,1))
dis_f<-0.99
reward_list<-c()
final_action_list<-list()
step_list<-c()
q_table<-list()
bi<-1

# target_qn<-model


for(i in 1:10000){
  total_r<-0 ## total reward
  episode_done<-0
  
  
  
  step<-1
  action_list<-NULL
  st<-c(1,1)
  
  dummy_act<-rep(1,length(action))
  while(episode_done==0){
    
    if(step >1){
      
      
      action_prob<-predict(model,list(t(cov_next_state),t(dummy_act)))[[1]]
      current_state<-next_state
      store_current_state<-(cov_next_state)
    }else{
      current_state<- next_where(c(1,1))
      action_prob<-predict(model,list(t(init_data),t(dummy_act)))[[1]]
      store_current_state<-init_data
      
      
    }
    action_index<-sample(1:4,1,prob=action_prob)
    (next_action<-  action[action_index])
    act_one_hot<-rep(0,4)
    act_one_hot[action_index]<-1
    
    action_list<-c(action_list,next_action)
    st<-move(st,next_action)
    state_index<-stm[st[1],st[2]]
    
    
    next_state<-next_where(state_index)
    re_ep<-return_reward(next_state,current_state) ## get a reward and Whether the episode ends for action(next state)
    
    cov_next_state<-convert_coord(st)
    total_r<-total_r+re_ep[1]
    episode_done<-re_ep[2]
    step<-step+1
    
    
    value<-predict(value_network,t(store_current_state))
    next_value<-predict(value_network,t(cov_next_state))
    vtarget<-re_ep[1]+dis_f*next_value
    td_error<-re_ep[1]+dis_f*next_value-value
    if(episode_done){
      td_error<-re_ep[1]-value
      vtarget<-re_ep[1]
    }
    
    train_on_batch(value_network,t(store_current_state),vtarget)
    train_on_batch(model,list(t(store_current_state),t(act_one_hot)),list(t(dummy_act),t(td_error)))
    
    
    
    if(step == 200){
      cat("\n",i," epsode-",step) 
      step_list<-c(step_list,step)
      final_action_list[[i]]<-action_list
      reward_list<-c(reward_list,total_r)

      ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
      
      break;
    }
    
    if(episode_done==1){
      
      cat("\n",i," epsode-",step) 
    
      step_list<-c(step_list,step)
      final_action_list[[i]]<-action_list
      reward_list<-c(reward_list,total_r)
      ts.plot(reward_list,main=paste0((reward_list)[length(reward_list)],"-",step,"-",min(step_list)))
      break;
    }
  }
  
  
  
  
}

