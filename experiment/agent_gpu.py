import tensorflow as tf

if len(tf.config.list_physical_devices('GPU')) != 0:
    print('Using *GPU*')
else:
    print('Warning. No GPU Access')

MAX_MEMORY = tf.constant(2**15, dtype=tf.int32) # ~ 32000
MIN_DEQUEUE = tf.constant(8, dtype=tf.int32)

class Model(tf.keras.Model):
    def __init__(self, n_outputs, n_inputs, units=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden1 = tf.keras.layers.Dense(units, activation='relu', input_shape=(n_inputs,))
        self.output_ = tf.keras.layers.Dense(n_outputs)
    
    def call(self, input_):
        x = self.hidden1(input_)
        x = self.output_(x)
        
        return x
        
class Agent:
    
    def __init__(self, n_actions=3, n_inputs=33, batch_size=1024, lr=0.01, gamma=.99, epsilonl=100) -> None:
        self.lr = tf.constant(lr, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        self.epsilonl = tf.constant(epsilonl, dtype=tf.int32)
        self.epsilon = tf.Variable(0)
        self.n_games = tf.Variable(0)
        self.n_actions = tf.constant(n_actions, dtype=tf.int32)
        self.test_mode = tf.Variable(False)
        
        self.model = Model(n_outputs=n_actions, n_inputs=n_inputs)
        self.optimizer = tf.optimizers.Adam(lr)
        self.loss_fn = tf.losses.mean_squared_error
        
        self.memory = tf.queue.RandomShuffleQueue(
            capacity=MAX_MEMORY,
            min_after_dequeue=MIN_DEQUEUE,
            dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.bool],
            shapes=[(1,n_inputs), (1,1), (1,1), (1,n_inputs), (1,1), (1,1)]
        )
    
    def predict(self, state) -> tf.Tensor:
        state = tf.cast(tf.constant([state]), tf.float32)
        
        proba = tf.random.uniform((1,), minval=0, maxval=self.epsilonl, dtype=tf.int32)
        
        self.epsilon.assign(tf.subtract(self.epsilonl, self.n_games))
        
        if proba < self.epsilon and not self.test_mode:
            return tf.random.uniform((1,), minval=0, maxval=self.n_actions, dtype=tf.int32)
        else:
            return tf.argmax(self.model(state), axis=1)
    
    def __call__(self, state):
        return self.predict(state)

    def remember(self, state, action, reward, next_state, done, truncate, many=False, preprocess=False):
        if self.__is_memory_safe():
            if preprocess:
                state = tf.cast(tf.constant([state]), dtype=tf.float32)
                action = tf.cast(tf.constant([[action]]), dtype=tf.float32)
                reward = tf.cast(tf.constant([[reward]]), dtype=tf.float32)
                next_state = tf.cast(tf.constant([next_state]), dtype=tf.float32)
                done = tf.cast(tf.constant([[done]]), dtype=tf.bool)
                truncate = tf.cast(tf.constant([[truncate]]), dtype=tf.bool)
            
            if not many:
                self.memory.enqueue((state, action, reward, next_state, done, truncate))
            else:
                self.memory.enqueue_many((state, action, reward, next_state, done, truncate))

    def train_step(self, states, actions, rewards, next_states, dones, truncates):
        next_q_values = self.model(next_states)
        
        runs = tf.subtract(tf.constant(1.0), tf.cast(tf.logical_or(dones, truncates), tf.float32))
        
        target = tf.add(rewards, tf.multiply(runs, tf.multiply(self.gamma, tf.reduce_max(next_q_values, axis=1))))
        
        target = tf.reshape(target, (-1, 1))
        
        mask = tf.one_hot(tf.cast(actions, tf.int32), self.n_actions)
        
        with tf.GradientTape() as tape:
            
            all_q_values = self.model(states)
            
            Q_values = tf.reduce_sum(tf.multiply(all_q_values, mask), axis=-1, keepdims=True)
            
            loss = tf.reduce_mean(self.loss_fn(target, Q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def remember_and_train_short(self, state, action, reward, next_state, done, truncate):
        state = tf.cast(tf.constant([state]), dtype=tf.float32)
        action = tf.cast(tf.constant([[action]]), dtype=tf.float32)
        reward = tf.cast(tf.constant([[reward]]), dtype=tf.float32)
        next_state = tf.cast(tf.constant([next_state]), dtype=tf.float32)
        done = tf.cast(tf.constant([[done]]), dtype=tf.bool)
        truncate = tf.cast(tf.constant([[truncate]]), dtype=tf.bool)
        
        self.remember(state, action, reward, next_state, done, truncate, preprocess=False, many=False)
        self.train_short(state, action, reward, next_state, done, truncate)
    
    def train_short(self, state, action, reward, next_state, done, truncate):
        self.train_step(state, action, reward, next_state, done, truncate)
    
    def train_long(self):
        
        differences = tf.subtract(self.memory.size(), MIN_DEQUEUE)
        
        if differences > self.batch_size:
            mini_sample = self.memory.dequeue_many(self.batch_size)
            
            self.train_step(*mini_sample)
            
        elif differences > 0:
            mini_sample = self.memory.dequeue_many(differences)
        
            self.train_step(*mini_sample)

    def __is_memory_safe(self):
        if self.memory.size()+MIN_DEQUEUE > MAX_MEMORY:
            return tf.constant(False)
        return tf.constant(True)
    
    def when_episode_done(self):
        self.n_games.assign_add(1)