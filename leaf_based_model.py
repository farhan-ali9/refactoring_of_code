import json
import os
import os.path
from datetime import datetime
import numpy as np
import pandas as pd

class LeafBasedModel:
    # path where to find the sentiment tables (earnings and complete)
    def __init__(self, path_features_folder, path_sentiment_folder, path_company_names, path_performance, end_date):
        self.path_companies = path_company_names
        self.path_pickle = f"{path_features_folder}/df_stock_11bins_{{}}_all.pkl"
        self.path_csv_sentiment = f"{path_sentiment_folder}{{}}_freq_all.csv"
        self.path_performance_summary = f"{path_performance}performance_summary_values_returns.csv"
        self.path_performance_signal = f"{path_performance} signal_ {end_date}.json"
        self.data_all = {}
        self.include_feature_parameters()
        
        
    def include_feature_parameters(self):
        # Sentiment features
        self.feature_base_sentiment = [
            "pos_sent_ratio",
            "neg_sent_ratio",
            "relevance",
            "pos_sent_abs_ratio",
            "neg_sent_abs_ratio"
        ]
        self.feature_base_sentiment_plot = [
            "Positive sentiment",
            "Positive share",
            "Negative sentiment",
            "Negative share",
            "News counts",
            "News relevance",
        ]
        # Technical features
        self.feature_base_tech = [
            "return past 7 days",
            "return past 14 days",
            "return past 30 days",
            "return past 60 days",
            "return past 90 days",
            "return past 180 days",
            "return past 365 days",
            "return past 14 days AVERAGE",
            "return past 30 days AVERAGE",
        ]
        self.feature_base_tech_plot = [
            "Market return past 14 days",
            "Market return past 30 days",
        ]
        # Sentiment past period
        self.periods = [30]

        # Features for model plotting
        self.features_model_plot = (
            self.feature_base_sentiment_plot + self.feature_base_tech_plot
        )
        # Decision tree parameters
        self.thresholds = [78, 83, 88]
        self.percentiles_return = [30, 70]

        # Update parameters
        self.update_period = 300
        self.initial_shift = 1
        self.target_periods = [30]
        self.target_periods_samples = 20

        # Trading parameters
        self.targets = [
            "return future " + str(i) + " days" for i in self.target_periods
        ]
        self.fee = 0.005
        self.fee_training = 0.01

        # Companies to remove
        self.comp_to_remove = ['Skan']

        # Filter parameters
        self.periods_filter = [[365, 30]]
        self.skip_performance = 15

        # Ensemble size
        self.ensemble_size_best = 15
        
    def compute_cap_weights(self):
    	# Read the companies data
    	df_companies = pd.read_excel(self.path_companies, sheet_name="Sheet1")
    	df_companies = df_companies.rename(columns={"Company identifier": "company"})

    	# Remove specified companies
    	df_companies = df_companies[~df_companies["company"].isin(self.comp_to_remove)]
    	firms = df_companies["company"].tolist()
    	weights_SPI = df_companies["weights_SPI"].tolist()
    	firms_plot = firms.copy()
    	nfirms = len(firms)
    	weights_SPI_normed = np.array(weights_SPI)
    	
    	# Calculate different cap weights
    	w_all = weights_SPI_normed.copy()
    	w_large = np.where(w_all < 4, 0, w_all)
    	w_medium = np.where(np.logical_or(w_all > 10, w_all <= 0.2), 0, w_all)
    	w_small_medium = np.where(w_all > 10, 0, w_all)
    	w_small = np.where(w_all > 0.9, 0, w_all)
    	
    	weights_names = ["Equal weights",
    	"All Cap",
    	"Large Cap",
    	"Medium Cap",
    	"Small Cap",
    	"Mid-Small Cap",
    	]
    	weights_values = np.zeros((nfirms, len(weights_names)))
    	weights_values[:, weights_names.index("Equal weights")] = np.ones(nfirms) / nfirms
    	weights_values[:, weights_names.index("All Cap")] = w_all
    	weights_values[:, weights_names.index("Large Cap")] = w_large
    	weights_values[:, weights_names.index("Medium Cap")] = w_medium
    	weights_values[:, weights_names.index("Small Cap")] = w_small
    	weights_values[:, weights_names.index("Mid-Small Cap")] = w_small_medium
    	
    	# Set attributes
    	self.weights_values = weights_values
    	self.firms = firms
    	self.firms_plot = firms_plot
    	self.weights_names = weights_names
    	self.weights_SPI = weights_SPI
    	print("Done with cap weights")

    def load_company_data(self, path_pickle, path_csv_sentiment):
    	if not (os.path.exists(path_pickle) and os.path.exists(path_csv_sentiment)):
    		return pd.DataFrame(None), True
    	df = pd.read_pickle(path_pickle)
    	df_sentiment = pd.read_csv(path_csv_sentiment).drop(columns="Unnamed: 0")
    	df_sentiment.index = pd.to_datetime(df_sentiment["date"])
    	df_merged = df.join(df_sentiment, how="inner")
    	df = df_merged.copy()
    	
    	# Filter out unwanted data
    	df = df[df["close"] > 0]
    	if len(df) == 0:
    		print("len(df) = 0 !")
    	return df, False

    def run_performance(self, returns=None, buy_signal=None, fee=0.005, start_value=100, skip=1):
    	ret = returns[::skip]
    	buy = buy_signal[::skip]
    
    	fees = np.zeros(len(buy))
    	fees[np.abs(np.diff(np.append(1, buy))) > 0.1] = fee
    	
    	buy_net = buy - fee
    	cumulative_returns = (ret[:-1] * buy_net[:-1] + 1).cumprod()
    	equity_cumulative = np.append(start_value, cumulative_returns * start_value)
    	return equity_cumulative
    
    def compute_threshold(self):
        print("starting with compute_threshold")
        length_series = {}
        data_all = {}
        company_to_remove = []
        first_valid_iteration = True
        max_length_over_all_companies = 0

        for company in self.firms:
            print("processing: " + company)
            df, skip_bool = self.load_company_data(
                self.path_pickle.format(company),
                self.path_csv_sentiment.format(company),
            )
            if skip_bool:
                print("skipping company process, company feature path does not exist.")
                company_to_remove.append(company)
                continue
            df = df[self.initial_shift :]
            length = len(df[df.index.year < 2020]) + int(
                len(df[df.index.year == 2021]) * 2 / 12
            )
            length_series[company] = length

            if length == 0:
                company_to_remove.append(company)

        max_length_over_all_companies = max(length_series.values())
        self.firms = [company for company in self.firms if company not in company_to_remove]

        for company in self.firms:
            print("processing: " + company)
            df, skip_bool = self.load_company_data(
                self.path_pickle.format(company),
                self.path_csv_sentiment.format(company),
            )
            if skip_bool:
                print("skipping company process, company feature path does not exist.")
                continue
            df = df[self.initial_shift :]

            if company in length_series:
                length = length_series[company]
                if length == 0:
                    continue
                update_starts = list(np.arange(0, length, self.update_period))
                update_starts = update_starts + [update_starts[-1] + self.update_period]
                update_period_loc = self.update_period
                num_updates = len(update_starts)
                rat_firm0 = length / self.update_period
                
                if first_valid_iteration:
                    update_starts_global = {company: update_starts}
                else:
                    update_period_loc = int(length / rat_firm0)
                    if update_period_loc == 0:
                        continue
                    update_starts = np.arange(0, length, update_period_loc)
                    update_starts = list(update_starts) + [update_starts[-1] + self.update_period]
                    update_starts = update_starts[:num_updates]
                    update_starts_global[company] = [
                        update_starts[i] + max_length_over_all_companies - length
                        for i in range(num_updates)
                    ]
                    update_starts_test = np.arange(
                        max_length_over_all_companies - length,
                        max_length_over_all_companies + self.update_period,
                        update_period_loc,
                    )
                    update_starts_global[company] = list(update_starts_test)[:num_updates]
                for per in self.periods:
                    for i in self.feature_base_sentiment:
                        features = str(i) + str(per)
                        for threshold in self.thresholds:
                            for update_start in update_starts:
                                if update_start == update_starts[-1]:
                                    end = len(df)
                                else:
                                    end = min(update_start + update_period_loc, len(df))
                                df20 = df.copy()[:end]
                                df_percentile = df20.copy()[:update_start]
                                buy = np.ones(len(df20))

                                if len(df_percentile[features].dropna()) > 1:
                                    aid = np.percentile(
                                        df_percentile[features].dropna().values, threshold
                                    )
                                    buy[df20[features] > aid] = 0
                                if update_start == update_starts[0]:
                                    data1 = pd.DataFrame(
                                        buy[update_start:end],
                                        columns=[company+ " buy "+ features+ " "+ str(threshold)],
                                        )
                                    data1.index = df20.index[update_start:end]
                                else:
                                    data2 = pd.DataFrame(
                                        buy[update_start:end],
                                        columns=[company+ " buy "+ features+ " "+ str(threshold)],
                                    )
                                    data2.index = df20.index[update_start:end]
                                    data1 = pd.concat([data1, data2])
                                if (threshold == self.thresholds[0]) & (
                                    i == self.feature_base_sentiment[0]
                                ):
                                    data = data1.copy()
                                data[company + " buy " + features + " " + str(threshold)] = data1

                for target_period in self.target_periods:
                    target = self.targets[self.target_periods.index(target_period)]
                    data[company + " " + target] = df[target]
                for i in self.feature_base_tech:
                    data[company + " " + i] = df[i]
                data[company + " close"] = df["close"]
                for per in self.periods:
                    for i in self.feature_base_sentiment:
                        features = str(i) + str(per)
                        data[company + " " + features] = df[features]

                mods = [s for s in self.feature_base_tech if str("AVERAGE") in s] + [
                    s for s in self.feature_base_tech if str("RSI") in s
                ]
                for per in self.periods:
                    for i in mods:
                        features = str(i)
                        for threshold in self.thresholds:
                            for update_start in update_starts:
                                if update_start == update_starts[-1]:
                                    end = len(df)
                                else:
                                    end = min(update_start + update_period_loc, len(df))

                                df20 = df.copy()[:end]
                                df_percentile = df20.copy()[:update_start]
                                buy = np.ones(len(df20))

                                if len(df_percentile[features].dropna()) > 1:
                                    aid = np.percentile(
                                        df_percentile[features].dropna().values,
                                        100 - threshold,
                                    )
                                    buy[df20[features] < aid] = 0
                                if update_start == update_starts[0]:
                                    data1 = pd.DataFrame(
                                        buy[update_start:end],
                                        columns=[company+ " buy "+ features+ " "+ str(threshold)],
                                    )
                                    data1.index = df20.index[update_start:end]
                                else:
                                    data2 = pd.DataFrame(
                                        buy[update_start:end],
                                        columns=[company+ " buy "+ features+ " "+ str(threshold)],
                                    )
                                    data2.index = df20.index[update_start:end]
                                    data1 = pd.concat([data1, data2])

                                data[company + " buy " + features + " " + str(threshold)] = data1
                data_all[company] = data
                first_valid_iteration = False
        self.data_all = data_all
        self.df = df
        self.update_starts_global = update_starts_global
        self.update_period_loc = update_period_loc
        self.length_series = length_series
        print("Done with compute threshold")
    
    def compute_historical_performance(self):
    	print("now starting in compute_historical_performance")
    	leaves = ["leaf1", "leaf2", "leaf3", "leaf4", "leaf5", "leaf6"]
    	vars_kpi = ["excess return ratio all", "excess return ratio weighted"]
    	
    	company = list(self.data_all.keys())[0]
    	all_columns_first_company = [
    		s for s in self.data_all[company].columns if str(company + " buy") in s
    		]
    	num_models = len(all_columns_first_company)
    	print("Number of models: ", num_models)
    	print("Models: ")
    	print(all_columns_first_company)
    	
    	update_starts = self.update_starts_global[list(self.data_all.keys())[0]]
    	
    	model_performance_all = np.zeros(
    	(
    	num_models,
    	len(leaves),
    	len(vars_kpi),
    	len(self.firms),
    	len(update_starts),
    	len(self.periods_filter),
    	)
    	)
    	for company in self.data_all:
    		for ifilter in range(len(self.periods_filter)):
    			self.data_all[company][company + " " + str(self.periods_filter[ifilter]) + " leaf index"] = 0
    	for company in self.data_all:
    		self.data_all[company][company + " update index"] = 0
    		update_starts = list(self.update_starts_global[company])
    		for update_start in update_starts:
    			ind0 = update_start
    			if update_start == update_starts[-1] or len(update_starts) <= 2:
    				ind1 = len(self.df) - 1
    				
    			else:
    				ind1 = min(update_start + update_starts[2] - update_starts[1],len(self.df) - 1,)
    			self.data_all[company][company + " update index"][ind0:ind1] = update_starts.index(update_start)
    			self.data_all[company][company + " update index"][-1] = np.max(
    			self.data_all[company][company + " update index"].values
    			)
    	for ifilter in range(len(self.periods_filter)):
    		features_filter = ["return past " + str(i) + " days" for i in self.periods_filter[ifilter]
    		]
    		for company in self.data_all:
    			update_starts = list(self.update_starts_global[company])
    			length = self.length_series[company]
    			start_data = max(0, int(update_starts[0]))
    			cols_company = [s for s in self.data_all[company].columns if company in s]
    			list_models = [s for s in self.data_all[company].columns if str(company + " buy") in s]
    			df_col = self.data_all[company][cols_company]
    			for update_start in update_starts:
    				end = len(df_col) if update_start == update_starts[-1] else int(min(update_start + self.update_period_loc, len(df_col)))
    				df = df_col.copy()[start_data:end]
    				df_percentile = df.copy()[start_data:update_start]
    				# compute percentiles needed for decision tree
    				aid1 = np.percentile(df_percentile[company + " " + features_filter[1]].dropna(), self.percentiles_return[0]) if len(df_percentile[company + " " + features_filter[1]].dropna()) > 1 else 0
    				aid2 = np.percentile(df_percentile[company + " " + features_filter[1]].dropna(), self.percentiles_return[1]) if len(df_percentile[company + " " + features_filter[1]].dropna()) > 1 else 0
    				ret365 = df[company + " " + features_filter[0]]
    				ret30 = df[company + " " + features_filter[1]]
    				ret_365_30 = (ret365 - ret30) / (1 + ret30)
    				for leaf in leaves:
    					mask = np.zeros(len(df), dtype=bool)
    					if leaf == "leaf1":
    						mask = (ret365 >= 0) & (ret30 >= aid2)
    					elif leaf == "leaf2":
    						mask = (ret365 >= 0) & (ret30 >= aid1) & (ret30 < aid2)
    					elif leaf == "leaf3":
    						mask = (ret365 >= 0) & (ret30 < aid1)
    					elif leaf == "leaf4":
    						mask = ((ret365 < 0) & (ret30 >= aid2) & (df.index.year < 2020)) | ((ret365 < 0) & (ret30 >= aid2) & (df.index.year > 2020))
    					elif leaf == "leaf5":
    						mask = ((ret365 < 0) & (ret30 >= aid1) & (ret30 < aid2) & (df.index.year < 2020)) | ((ret365 < 0) & (ret30 >= aid1) & (ret30 < aid2) & (df.index.year > 2020))
    					elif leaf == "leaf6":
    						mask = ((ret365 < 0) & (ret30 < aid1) & (df.index.year < 2020)) | ((ret365 < 0) & (ret30 < aid1) & (df.index.year > 2020))
    						
    						df[company + " " + str(self.periods_filter[ifilter]) + " leaf index"][mask] = leaves.index(leaf)
    						df_leaf = df[mask][:update_start]
    						if len(df_leaf) > 1:
    							if (
    							update_start == update_starts[-1]
    							or len(update_starts) <= 2
    							):
    								mask2 = [False for _ in range(start_data)] + [
    								mask[i] for i in range(len(mask))
    								]
    								self.data_all[company].loc[mask2,company+ " "+ str(self.periods_filter[ifilter])+ " leaf index",] = leaves.index(leaf)



    							w = [0.8, 1.2]
    							length = int(len(df_leaf) / 2)
    							ret1 = df_leaf[company + " " + self.targets[self.target_periods.index(30)]][:length]
    							ret2 = df_leaf[company + " " + self.targets[self.target_periods.index(30)]][length:]
    							num_ret1 = len(ret1[:: self.skip_performance])
    							num_ret2 = len(ret2[:: self.skip_performance])
    							
    							hold_leaf1 = self.run_performance(returns=ret1, buy_signal=np.ones(len(ret1)), skip=self.skip_performance)
    							hold_tot1 = hold_leaf1[-1] ** (12 / num_ret1)
    							hold_leaf2 = self.run_performance(returns=ret2, buy_signal=np.ones(len(ret2)), skip=self.skip_performance)
    							hold_tot2 = hold_leaf2[-1] ** (12 / num_ret2)
    							hold_tot = hold_tot1 * hold_tot2
    							ratio_model = []
    							ratio_model_weighted = []
    							count_negative_hold = len(df_leaf[df_leaf[company + " " + self.targets[self.target_periods.index(30)]] < 0][:: self.skip_performance])
    							for i in list_models:
    								perf1 = self.run_performance(returns=ret1, buy_signal=df_leaf[i][:length], fee=self.fee_training, skip=self.skip_performance)
    								perf2 = self.run_performance(returns=ret2, buy_signal=df_leaf[i][length:], fee=self.fee_training, skip=self.skip_performance)
    								perf_tot1 = perf1[-1] ** (12 / num_ret1)
    								perf_tot2 = perf2[-1] ** (12 / num_ret2)
    								perf_tot = perf_tot1 * perf_tot2
    								ratio_model_weighted.append((perf_tot1 / hold_tot1) ** w[0] * (perf_tot2 / hold_tot2) ** w[1])
    								ratio_model.append(perf_tot / hold_tot)
    							model_performance_all[:, leaves.index(leaf), vars_kpi.index("excess return ratio all"),self.firms.index(company), update_starts.index(update_start), ifilter] = ratio_model
    							model_performance_all[:, leaves.index(leaf), vars_kpi.index("excess return ratio weighted"), self.firms.index(company), update_starts.index(update_start), ifilter] = ratio_model_weighted
    	self.num_models = num_models
    	self.leaves = leaves
    	
    	self.ind_best_models = np.zeros(
    	(
    	self.ensemble_size_best,
    	len(self.leaves),
    	len(self.firms),
    	len(
    		self.update_starts_global[
    		max(
    			self.update_starts_global,
    			key=lambda x: len(set(self.update_starts_global[x])),
    		)
    		]
    		),
    	len(self.periods_filter),
    	)
    	)
    	self.model_performance_all = model_performance_all
    	self.vars_kpi = vars_kpi
    	print("Done with performance computation")
    def rank_models(self):
    	print("Done with model ranking")
    	kpi_vars = ["av performance", "av rank performance"]
    	kpi_num = np.zeros((self.num_models, len(self.leaves), len(kpi_vars)))
    	list_models = []
    	
    	for company in self.data_all:
    		list_models += [ s for s in self.data_all[company].columns
    		if str(self.firms[-1] + " buy") in s
    		]
    	for ifilter in range(len(self.periods_filter)):
    		for leaf in self.leaves:
    			rank_perf = np.zeros(self.num_models)
    			perf_av = np.zeros(self.num_models)
    			for company in self.firms:
    				if company not in self.update_starts_global:
    					continue
    				update_starts = list(self.update_starts_global[company])
    				for update_start in update_starts:
    					perf = pd.DataFrame(
    					self.model_performance_all[
    					:,
    					self.leaves.index(leaf),
    					self.vars_kpi.index("excess return ratio all"),
    					self.firms.index(company),
    					update_starts.index(update_start),
    					ifilter,
    					]
    					)
    					
    					perf_av = perf_av + perf[0].values
    					ind = list(perf.sort_values(by=0, ascending=False).index)
    					rank = [ind.index(i) for i in range(len(list_models))]
    					rank_perf = rank_perf + rank
    				ind_to_store = ind[: self.ensemble_size_best] if len(ind) >= self.ensemble_size_best else ind[:]
    				self.ind_best_models[
    				: len(ind_to_store),
    				self.leaves.index(leaf),
    				self.firms.index(company),
    				update_starts.index(update_start),
    				ifilter,
    				] = ind_to_store
    			kpi_num[:, self.leaves.index(leaf), kpi_vars.index("av performance")] = perf_av / len(self.firms)
    			kpi_num[:, self.leaves.index(leaf), kpi_vars.index("av rank performance")] = rank_perf / len(self.firms)
    	print("Done with model ranking")


    def performance_test(self):
        print("Starting with performance_test")
        data_all_performance_test = {}

        for company in self.data_all:
            data_all_performance_test[company] = pd.concat([self.data_all[company][::self.target_periods_samples], self.data_all[company][-1:]])

        ensembles = [8]
        shares = [40]
        leaf_selects = ["all", "bear", "bull", "only leaf 5-6", "only leaf 6"]
        
        for company in self.firms:
            if company not in data_all_performance_test:
                continue
            close = data_all_performance_test[company][company + " close"]
            ret = [(close[i] / close[i - 1]) - 1 for i in range(1, len(close))]
            ret_net = np.append(ret, 0)
            data_all_performance_test[company][company + " " + "return future tailor days"] = ret_net

        for ifilter in range(len(self.periods_filter)):
            for company in self.firms:
                if company not in data_all_performance_test:
                    continue
                index_leaf = data_all_performance_test[company][company + " " + str(self.periods_filter[ifilter]) + " leaf index"]
                list_models_loc = [s for s in data_all_performance_test[company].columns if str(company + " buy") in s]

                for ens in ensembles:
                    for share in shares:
                        buy_cum = []
                        for i in range(len(index_leaf)):
                            periods = self.update_starts_global[company]
                            i_all = int(i * self.target_periods_samples)
                            if i_all < periods[0]:
                                buy = 0
                            else:
                                index_period = data_all_performance_test[company][company + " update index"][i]
                                buy = 0
                                list_index = self.ind_best_models[:ens, index_leaf[i], self.firms.index(company), index_period, ifilter].astype(int)
                                for kk in list_index:
                                    buy = buy + data_all_performance_test[company][list_models_loc[kk]][i]
                            buy_cum.append(buy)

                        buy_binary = np.array(buy_cum)
                        for leaf_select in leaf_selects:
                            buy_binary_loc = buy_binary.copy()
                            leaf_index = data_all_performance_test[company][company + " " + str(self.periods_filter[ifilter]) + " leaf index"]
                            if leaf_select == "bear":
                                buy_binary_loc[leaf_index <= 2] = 1
                            elif leaf_select == "bull":
                                buy_binary_loc[leaf_index >= 3] = 1
                            elif leaf_select == "only leaf 5-6":
                                buy_binary_loc[leaf_index <= 3] = 1
                            elif leaf_select == "only leaf 6":
                                buy_binary_loc[leaf_index <= 4] = 1
                            data_all_performance_test[company][
                                "buy "
                                + leaf_select
                                + " "
                                + str(self.periods_filter[ifilter])
                                + " "
                                + str(ens)
                                + " "
                                + str(share)
                                + " "
                                + company
                            ] = buy_binary_loc
                            data_all_performance_test[company][
                                "eq "
                                + leaf_select
                                + " "
                                + str(self.periods_filter[ifilter])
                                + " "
                                + str(ens)
                                + " "
                                + str(share)
                                + " "
                                + company
                            ] = self.run_performance(
                                returns=data_all_performance_test[company][
                                    company + " " + "return future tailor days"
                                ].fillna(0),
                                fee=self.fee,
                                buy_signal=buy_binary_loc,
                            )

                        # hold strategy is always the same
                        data_all_performance_test[company][
                            "buy hold " + company
                        ] = np.ones(len(data_all_performance_test[company]))
                        data_all_performance_test[company][
                            "eq hold " + company
                        ] = self.run_performance(
                            returns=data_all_performance_test[company][
                                company + " " + "return future tailor days"
                            ].fillna(0),
                            buy_signal=np.ones(len(buy_binary)),
                        )

        models = []
        for ifilter in range(len(self.periods_filter)):
            for leaf_select in leaf_selects:
                for ens in ensembles:
                    for share in shares:
                        models.append(leaf_select + " " + str(self.periods_filter[ifilter]) + " " + str(ens) + " " + str(share))
        models.append("hold")

        for mod in models:
            for company in self.firms:
                if company not in data_all_performance_test:
                    continue
                variables = ["eq " + mod + " " + company]
                buys = ["buy " + mod + " " + company]

                data_all_performance_test[company]["total " + mod] = data_all_performance_test[company][variables].mean(axis=1)
                data_all_performance_test[company]["total cash " + mod] = np.mean(
                    (np.array(data_all_performance_test[company][variables]) * np.array((1 - data_all_performance_test[company][buys]))),
                    axis=1,
                )

        self.df_performance_test = data_all_performance_test
        self.column_to_extract = "buy bear " + str(self.periods_filter[0]) + " " + str(ensembles[0]) + " " + str(shares[0]) + " "
        print("Done with performance test")

        return data_all_performance_test

    @staticmethod
    def create_monthly_signal(signal_dict):
        # Initialize the resulting monthly signal dictionary
        monthly_signal_dict = {}
        
        # Iterate over each company in the signal_dict
        for company, date_signals in signal_dict.items():
            # Initialize the company's monthly signal sub-dictionary
            monthly_signal_dict[company] = {}
            
            # Create a dictionary to group dates by year-month
            date_groups = {}
            for date, signal in date_signals.items():
                date_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                year_month_string = date_object.strftime("%Y-%m")
                
                if year_month_string not in date_groups:
                    date_groups[year_month_string] = []
                date_groups[year_month_string].append((date, signal))
            
            # Calculate monthly signals for each year-month group
            for year_month, date_signal_pairs in date_groups.items():
                monthly_signal = 1.0
                for date, signal in date_signal_pairs:
                    if signal == 0:
                        monthly_signal = 0
                
                # Store the calculated monthly signal in the result dictionary
                monthly_signal_dict[company][year_month] = monthly_signal
        
        return monthly_signal_dict
    def store(self):
        # Extract signals and store in JSON
        signal_dict = {}
        for company in self.df_performance_test:
            column_to_extract_store = self.column_to_extract + company
            tmp = self.df_performance_test[company].to_dict()
            tmp_feature = {}
            for feature_time in tmp[column_to_extract_store]:
                tmp_feature[str(feature_time)] = tmp[column_to_extract_store][feature_time]
            signal_dict[company] = tmp_feature
        
        # Save the signal_dict as a JSON file
        with open(self.path_performance_signal, "w") as json_file:
            json.dump(signal_dict, json_file, indent=2)
        
        # Calculate and store monthly signals
        monthly_signal_dict = self.create_monthly_signal(signal_dict)
        list_times = [date for company_signals in monthly_signal_dict.values() for date in company_signals.keys()]
        unique_list_times = sorted(list(set(list_times)))
        
        # Prepare data for DataFrame
        signal_dict_for_df = {"date": [str(date) for date in unique_list_times]}
        for company in self.df_performance_test:
            signal_dict_for_df[company] = []
            past_value = None
            for date in unique_list_times:
                if date in monthly_signal_dict[company]:
                    signal_dict_for_df[company].append(monthly_signal_dict[company][date])
                    past_value = monthly_signal_dict[company][date]
                elif past_value == 0 or past_value == 1:
                    signal_dict_for_df[company].append(past_value)
                else:
                    signal_dict_for_df[company].append("")
        
        # Create a DataFrame from the signal data and save as an Excel file
        signal_df_to_store = pd.DataFrame.from_dict(signal_dict_for_df)
        signal_df_to_store.to_excel(self.path_performance_signal.replace(".json", ".xlsx"))
        
        # Calculate and store performance summary
        var_table = [
            "weight", "return 20 years model", "return 20 years hold",
            "excess return 20 years", "yearly return model", "yearly return hold",
            "yearly excess return", "tracking error", "information ratio",
            "max drawdown model", "max drawdown hold"
        ]
        table_val = np.zeros((len(self.firms), len(var_table)))
        
        fontsize = 16
        ens = 8
        share = 40
        mod = "bear [365, 30] "
        
        ratio = []
        return_model = []
        return_hold = []
        
        for company in self.firms:
            if company not in self.df_performance_test:
                continue
            ind = self.firms.index(company)
            ret_20_model = self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"].dropna().iloc[-1] - 100
            ret_20_hold = self.df_performance_test[company][f"eq hold {company}"].iloc[-1] - 100
            excess_ret_20 = ret_20_model - ret_20_hold
            ann_ret_model = (((ret_20_model + 100) / 100) ** (1 / 20) - 1) * 100
            ann_ret_hold = (((ret_20_hold + 100) / 100) ** (1 / 20) - 1) * 100
            ann_excess_ret = ann_ret_model - ann_ret_hold

            list_years = self.df_performance_test[company].index.year.unique()
            
            ret_model = []
            eq_model = self.df_performance_test[company][f"eq {mod}{ens} {share} {company}"]
            for i in list_years:
                eq_1 = eq_model[self.df_performance_test[company].index.year == i]
                eq_2 = eq_model[self.df_performance_test[company].index.year == i - 1]
                if not eq_1.empty and not eq_2.empty:
                    ret_model.append(((eq_1.iloc[-1] / eq_2.iloc[-1]) - 1) * 100)
            
            ret_hold = []
            eq_hold = self.df_performance_test[company][f"eq hold {company}"]
            for i in list_years:
                eq_1 = eq_hold[self.df_performance_test[company].index.year == i]
                eq_2 = eq_hold[self.df_performance_test[company].index.year == i - 1]
                if not eq_1.empty and not eq_2.empty:
                    ret_hold.append(((eq_1.iloc[-1] / eq_2.iloc[-1]) - 1) * 100)
            
            tracking_error = np.std(np.array(ret_model) - np.array(ret_hold))
            information_ratio = ann_excess_ret / tracking_error
            
            eq_model_values = eq_model.values
            eq_hold_values = self.df_performance_test[company][f"eq hold {company}"].values
            peak_cum = np.maximum.accumulate(eq_model_values)
            drawdown_model = np.min(100 * (eq_model_values - peak_cum) / peak_cum)
            
            peak_cum_hold = np.maximum.accumulate(eq_hold_values)
            drawdown_hold = np.min(100 * (eq_hold_values - peak_cum_hold) / peak_cum_hold)
            
            table_val[ind, var_table.index("return 20 years model")] = np.int64(ret_20_model)
            table_val[ind, var_table.index("return 20 years hold")] = np.int64(ret_20_hold)
            table_val[ind, var_table.index("excess return 20 years")] = np.int64(excess_ret_20)
            table_val[ind, var_table.index("yearly return model")] = round(ann_ret_model, 1)
            table_val[ind, var_table.index("yearly return hold")] = round(ann_ret_hold, 1)
            table_val[ind, var_table.index("yearly excess return")] = round(ann_excess_ret, 1)
            table_val[ind, var_table.index("tracking error")] = round(tracking_error, 1)
            table_val[ind, var_table.index("information ratio")] = round(information_ratio, 2)
            table_val[ind, var_table.index("max drawdown model")] = int(drawdown_model)
            table_val[ind, var_table.index("max drawdown hold")] = int(drawdown_hold)
        
        w = self.weights_SPI / np.sum(self.weights_SPI) * 100
        table_val[:, var_table.index("weight")] = [round(w[i], 3) for i in range(len(self.firms))]
        df_table = pd.DataFrame(table_val, columns=var_table)
        df_table.index = self.firms
        
        df_table.to_csv(self.path_performance_summary)
        print("Saved file: " + self.path_performance_summary)
        # Read the CSV file into a DataFrame
        saved_data = pd.read_csv(self.path_performance_summary, index_col=0)
        # Print the DataFrame
        print(saved_data)



if __name__ == "__main__":
    # adapt line below with the path where the local files for testing are stored.
    base_folder = "//home/farhan/FIverr project/unit_tests/unit_tests"
    model_config = {
        "path_performance": base_folder + "/performance/",
        "path_features_folder": base_folder + "/price_data/",
        "path_sentiment_folder": base_folder + "/sent_write_path_tables_2022-12-31/",
        "path_company_names": base_folder
        + "/mapping_companies_unit_tests.xlsx",
    }
    LBM = LeafBasedModel(
        model_config["path_features_folder"],
        model_config["path_sentiment_folder"],
        model_config["path_company_names"],
        model_config["path_performance"],
        "end_date",
    )
    LBM.compute_cap_weights()
    LBM.compute_threshold()
    LBM.compute_historical_performance()
    LBM.rank_models()
    df_companies_test = LBM.performance_test()
    LBM.store()
