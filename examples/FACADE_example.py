from pipeline import Pipeline,Scenario, Attacker,model_wrapper

def main():
     # initialize Atacker, which specifies access rights
    training_data_access = True
    dev_data_access = False
    test_data_access = False
    model_access = False
    output_access = 0
    myattacker = Attacker(training_data_access,dev_data_access,test_data_access,model_access,output_access)

    # initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target,myattacker)

    model_wrapper = Pipeline(myscenario,train_data,dev_data,test_data,model,training_process,device).get_object()
    test(model_wrapper, device,validation_sampler, 5,vocab)

if __name__ == "__main__":
    main()