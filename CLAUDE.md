Hi Claude, 
You're a wonderful datascientist and i need you to apply you skills and statistical methods to a new problem

the problem I need you to model in pytorch is predicting wait time for car collection.w

Factors that should go in are to the data seeding are: 
 location/dealer,
 driver,
 time of day,
 checklist length, 
whether its an ev and may need charge or the car needs fuel, (charging takes hours - unlike just fuel)  
whether the driver needs lunch etc. 

These are mostly known by the company for each sample, but there are factors impossible to model, so we should use an  MCMC method.

Task: 
 Please create a ML model tuning framework where a table of pickups with sampled weight times is fitted to by MCMC.

 Add visualisations for initial of posterior distributions and log to wandb.


 
Next Steps: 
Ultimately, this is to be a single component in a wider framework - doing the same steps for predicting journey travel times, and selecting routes by reducing risk. 

We will also eventually add in an optimised selector, finding the best return route for a driver, based on public transport or other journeys- which will be a risk/reward trade off between booking early on public transport to minimize cost, balanced by the potential for waiting for a free ride of another car going past, (if that doesnt affect the risk on that journey>!) 

So there's a lot of complex data flows and analytics to perform. 

Build a data dashboard with a simulated model, showing the predicted optimums and with the option to show how these algorithms behave with more or less requested journeys. 


When finished: 

Dockerise and and github actions for automated build and documentation of the system. 

If you get there, the bonus task is to analyse whether RL approaches for booking travel, waiting or selecting available drivers improves on the statistical approach or not. 


