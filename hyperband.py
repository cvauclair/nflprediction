
# Hyperband algorithm
# Link: http://people.eecs.berkeley.edu/~kjamieson/hyperband.html
# Paper: https://arxiv.org/pdf/1603.06560.pdf
# ds_rate: downsampling rate
# max_epochs: the maximum number of epochs for which the algorithm will train a model
# random_model_function: a function which returns a random model
# evaluation_function: a function of the form "f(model)" which returns the loss of a model
def hyperband(ds_rate, max_epochs, random_model_function, evaluation_function):
	log_ds_rate = lambda x: math.log(x)/math.log(self.ds_rate)

	# Number of unique executions of Successive Halving (minus one)
	s_max = int(log_ds_rate(self.max_epochs))          

	# Total number of iterations (without reuse) per execution of 
	# Succesive Halving (n,r)
	B = (s_max + 1)*self.max_epochs  

	#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
	for s in reversed(range(0, s_max + 1)):
	    print("-------------- s = {} ---------------".format(s))
	    # Initial number of configurations
	    n = int(math.ceil(int(B/self.max_epochs/(s + 1)) * self.ds_rate**s)) 
	    
	    # Initial number of iterations to run configurations for
	    r = max_epochs * ds_rate**(-s)

	    #### Begin Finite Horizon Successive Halving with (n,r)
	    models = [random_model_function()] * n
	    for i in range(s+1):
	        # Run each of the models for num_epochs and keep best num_models/ds_rate
	        num_models = int(n * ds_rate**(-i))
	        num_epochs = int(r * ds_rate**(i))
	        print("num_epochs = {}, num_configs = {}".format(num_epochs, num_configs))
	        
	        losses = []
	        for model in models:


	        # Set losses
	        for i, config in enumerate(configs):
	            config['num_epochs'] = num_epochs
	            config['loss'] = losses[i]
	            
	            # Check if update best config
	            self.__update_best(config) 

	        configs = self.__k_best(configs, losses, int(num_configs/self.ds_rate))
	        print("")
	    #### End Finite Horizon Successive Halving with (n,r)