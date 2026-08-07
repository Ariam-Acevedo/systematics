file_path = "energy_charge_events_streamed.pkl"
files_path_up = "birks_a_1_sig_up.pkl"
files_path_down = "birks_a_1_sig_dn.pkl"
files_nominal = "/global/homes/a/ariam/hurricane method/nom_files_MR6_2.pkl"
birks_a_nominal = 0.8


# num_events = 4000
# range_e = (0, 8)
# num_bins = 19*4
# density = False
def analysis(num_events,fixed_events,  range_e, num_bins, density,log=False):
    rng = np.random.default_rng(42)   # fixed seed for reproducibility


    
    #files procesing

    #Snowstorm files
    with open(file_path, "rb") as f:
        data_files = pickle.load(f)
    
    indices = rng.choice(len(data_files), size=num_events, replace=False)
    indices.sort()   
    
    data = ak.Array([data_files[i]["energyHits"] for i in indices])
    parameter_values =  ak.Array([np.repeat(data_files[i]["birks_a"], len(data_files[i]["energyHits"])) for i in indices])

    #fixed files 1 sigtma up
    with open(files_path_up, 'rb') as f:
            data_files_up = pickle.load(f)
            data_up=ak.Array([event["energyHits"] for event in data_files_up[:fixed_events]])

    #fixed files 1 sigma down
    with open(files_path_down, 'rb') as f:
            data_files_down = pickle.load(f)
            data_down=ak.Array([event["energyHits"] for event in data_files_down[:fixed_events]])

    #Nominal mini run 6.2
    with open(files_nominal, 'rb') as f:
            data_files_nominal = pickle.load(f)
            data_nominal=ak.Array([event["energyHits"] for event in data_files_nominal[:fixed_events]])






    print(num_events, "snowstorm events processed. ", fixed_events, " events procesed for each ordinary pickle file.")
    print("Mean of birks_a:", np.mean(parameter_values))
    print("Std of birks_a:", np.std(parameter_values))

    
    #stattistics
    mu = np.mean(parameter_values)
    sigma = np.std(parameter_values)


    ######################################### snowstorm calculation ######################################

    #mask done at event level
    mask_up = parameter_values[:, 0] > mu
    mask_down = parameter_values[:, 0] < mu

    #masking data based on birks_a values
    values_up=data[mask_up]
    values_down=data[mask_down]

    values_up_flat = ak.flatten(values_up)      #flattened ak array
    values_down_flat = ak.flatten(values_down)

    # Histograms
    numpy_nominal, bins = np.histogram(np.array(ak.flatten(data)), bins=num_bins, range =range_e, density=density)
    numpy_values_up,_ = np.histogram(np.array(values_up_flat), bins=num_bins, range =range_e, density=density)
    numpy_values_down, _ = np.histogram(np.array(values_down_flat), bins=num_bins, range =range_e, density=density)
    fixed_values_up = np.histogram(np.array(ak.flatten(data_up)), bins=num_bins, range=range_e, density=density)
    fixed_values_down = np.histogram(np.array(ak.flatten(data_down)), bins=num_bins, range=range_e, density=density)
    nominal_6_2_light = np.histogram(np.array(ak.flatten(data_nominal)), bins=num_bins, range =range_e, density=density)


    #gradient calculation
    grad_a = np.sqrt(np.pi/2)*(numpy_values_up - numpy_values_down)*(1.0/sigma)

    #ss 1 sig calculation
    snowstorm_up=nominal_6_2_light[0]+grad_a*sigma
    snowstorm_down=nominal_6_2_light[0]-grad_a*sigma

    #######################################################################################################################


    print(len(parameter_values[mask_up]), f"events with birks_a > {birks_a_nominal}")
    print(len(parameter_values[mask_down]), f"events with birks_a < {birks_a_nominal}")
    print("Number of events with more than 20000 hits up evemts:", np.sum(ak.num(data[mask_up])>20000))
    print("Number of events with more than 20000 hits down events:", np.sum(ak.num(data[mask_down])>20000))
    print("Number of hits  up evemts:", len(values_up_flat))
    print("Number of hits down events:", len(values_down_flat))
    
    ### ploting 
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    ax00 = axs[0, 0]
    ax01 = axs[0, 1]
    ax10 = axs[0, 2]
    ax11 = axs[1, 1]
    ax20 = axs[1, 0]
    ax21 = axs[1, 2]

    if density:
        y_axis_label = "Density"
    else:
        y_axis_label = "Counts"

    textstr = '\n'.join((
    r'$\mu=%.4f$' % (mu, ),
    r'$\sigma=%.4f$' % (sigma, )))

    #Distribution of parameter values on hit level
    ax00.hist(ak.flatten(parameter_values[mask_up]), bins=15, alpha=0.5, label=f"birks_a > {birks_a_nominal}")
    ax00.hist(ak.flatten(parameter_values[mask_down]), bins=15, alpha=0.5, label=f"birks_a < {birks_a_nominal}")
    ax00.text(0.05, 0.95, textstr, transform=ax00.transAxes, fontsize=12, verticalalignment='top')
    ax00.set_xlabel("birks_a")
    ax00.set_ylabel("Frequency")
    ax00.set_title("Distribution of birks_a")
    ax00.legend()



    #Initial separation
    
    ax01.stairs(numpy_values_up, bins, label="birks_a > 0.8")
    ax01.stairs(numpy_values_down, bins, label="birks_a < 0.8")
    ax01.stairs(numpy_nominal if density else numpy_nominal/2, bins, label="All Events")
    ax01.set_xlim(0,2)
    ax01.set_yscale("log" if log else "linear")
    ax01.set_xlabel("Bin")
    ax01.set_ylabel(y_axis_label)
    ax01.set_title("Distribution of Energy Hits")
    ax01.legend()

    
    #Gradient
    ax10.stairs(grad_a, bins, label="grad_a")
    ax10.set_xlabel("Bin")
    ax10.set_ylabel("Gradient")
    ax10.set_title("Distribution of Gradient")
    ax10.set_xlim(0,2)
    ax10.legend()
    
    #Snowstorm plots    
    ax11.stairs(nominal_6_2_light[0], bins, label="Nominal", color="black", alpha=0.7)
    ax11.stairs(snowstorm_up, bins, label="Nominal + 1 sigma", color="tab:blue", alpha=0.7)
    ax11.stairs(snowstorm_down, bins, label="Nominal - 1 sigma", color="tab:orange", alpha=0.7)
    ax11.set_xlim(0,2)
    ax11.set_title("Snowstorm Approximation")
    ax11.set_xlabel("Energy")
    ax11.set_ylabel(y_axis_label)
    ax11.set_yscale("log" if log else "linear")
    ax11.legend()


    #Ordinary nominal and systematics 
    ax20.stairs((nominal_6_2_light[0] if not density else nominal_6_2_light[0]/2), nominal_6_2_light[1], label="Nominal 6.2 Light", color="black", alpha=0.7)
    ax20.stairs(fixed_values_up[0], fixed_values_up[1], label="Fixed Values + 1 sigma", color="tab:blue", alpha=0.7)
    ax20.stairs(fixed_values_down[0], fixed_values_down[1], label="Fixed Values - 1 sigma", color="tab:orange", alpha=0.7)
    ax20.set_xlim(0,2)
    ax20.set_yscale("log" if log else "linear")
    ax20.set_title("Reference Sample")
    ax20.set_xlabel("Energy")
    ax20.set_ylabel(y_axis_label)
    ax20.legend()


    #difference plotting
    eps = 0.001
    difference=snowstorm_up - snowstorm_down

    fixed_difference = fixed_values_up[0] - fixed_values_down[0]

    
    ax21.stairs(difference, bins,
                label="snowstorm difference")

    ax21.stairs(fixed_difference, bins,
                label="fixed difference")

    ax21.axhline(0.0, color="black", ls="--")

    ax21.set_xlim(0,2)
    ax21.set_ylim(np.array([difference.min(),fixed_difference.min()]).min()-eps,np.array([difference.max(),fixed_difference.max()]).max()+eps   )     # adjust if needed

    ax21.set_title("Difference to up-down")
    ax21.set_xlabel("Energy")
    ax21.set_ylabel("Difference")
    ax21.legend()

    fig.suptitle(f"Birks_a Diagnostic (N = {num_events})", fontsize=18)
    fraction_up = len(values_up_flat)/len(ak.flatten(data))
    fraction_down = len(values_down_flat)/len(ak.flatten(data))

    fig.text(
        0.02, 0.98,
        f"Events: {num_events}\n"
        f"Mean = {mu:.5f}\n"
        f"Std = {sigma:.5f}\n"
        f"Hit fraction up = {fraction_up:.3f}\n"
        f"Hit fraction down = {fraction_down:.3f}\n",
        # f"Hits number = {len(ak.flatten(data_up)) + len(ak.flatten(data_down))}\n",
        fontsize=10,
        va="top"
    )

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.show()

#Example run: 

analysis(num_events=10000, range_e=(0,8), num_bins=4*19, density=False,log=False, fixed_events=4000)

