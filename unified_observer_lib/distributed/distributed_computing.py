import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

class DistributedTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
        # Initialize Horovod
        hvd.init()
        
        # Pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
        # Wrap optimizer with Horovod Distributed Optimizer
        self.optimizer = hvd.DistributedOptimizer(self.optimizer)

    @tf.function
    def distributed_train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = tf.keras.losses.mean_squared_error(targets, predictions)
        
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss

    def train(self, dataset, epochs):
        # Horovod: adjust number of steps based on number of GPUs.
        dataset = dataset.shard(hvd.size(), hvd.rank())
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in dataset:
                inputs, targets = batch
                loss = self.distributed_train_step(inputs, targets)
                epoch_loss += loss
                num_batches += 1
            
            # Horovod: use broadcast to ensure consistent epoch_loss across all workers
            avg_loss = hvd.allreduce(epoch_loss / num_batches, name="avg_loss")
            
            # Horovod: broadcast metrics from rank 0 to all other processes.
            if hvd.rank() == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss}")

    def save_model(self, path):
        # Horovod: save model only on worker 0 to prevent conflicts between workers
        if hvd.rank() == 0:
            self.model.save(path)

class ParallelProcessor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or hvd.size()

    def parallel_map(self, func, data):
        # Simple parallel map implementation using Horovod
        shard_size = len(data) // self.num_workers
        start = hvd.rank() * shard_size
        end = start + shard_size if hvd.rank() < self.num_workers - 1 else len(data)
        
        local_results = [func(item) for item in data[start:end]]
        
        # Gather results from all workers
        all_results = hvd.allgather_object(local_results)
        
        # Flatten the list of results
        return [item for sublist in all_results for item in sublist]
