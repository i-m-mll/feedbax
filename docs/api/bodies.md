# Bodies

These are the highest-level single time step models included in Feedbax. They are compositions of neural networks, mechanical models, delayed channels, and other components. Typically they model a single step through an entire sensory-motor loop. 

A stereotypical example is 1) a forward pass of a neural network (AKA controller, agent, or policy) results in motor commands (AKA controls, or actions), which are 2) sent to a biomechanical model (AKA plant, or environment) whose state is updated by integrating some differential equations, and 3) sensory feedback about the state is returned so that it can bee passed to the neural network on the next iteration.

Currently, there is only one such model: [`SimpleFeedback`][feedbax.bodies.SimpleFeedback]. 

---

::: feedbax.bodies.SimpleFeedbackState
        
::: feedbax.bodies.SimpleFeedback


        