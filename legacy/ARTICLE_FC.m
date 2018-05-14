clear;
layer_name = {'res2a_branch2a', 'res2b_branch2a', 'res2c_branch2a', 'res3a_branch2a', 'res3b_branch2a', 'res3c_branch2a', 'res3d_branch2a', 'res4a_branch2a', 'res4b_branch2a', 'res4c_branch2a', 'res4d_branch2a', 'res4e_branch2a', 'res4f_branch2a', 'res5a_branch2a', 'res5b_branch2a', 'res5c_branch2a','avg_pool'};
objs_name = {'Black-Backpack','Black-Coat','Brown-Box','Camera-Box','Dark-blue-Box','Pink-Bottle','Shoe','Towel','White-Jar','Mult-Objs1','Mult-Objs2','Mult-Objs3'};
test_objs_blocks = 59;

TP_ratio = [1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,0.570000,0.725000,0.785000,0.765000,0.785000,0.735000,0.725000,0.785000,0.745000,0.850000,0.740000,0.750000,0.740000,0.600000,0.835000,0.735000,0.690000,1.000000,0.765000,1.000000,1.000000,0.750000,0.715000,0.745000,0.740000,0.755000,0.675000,0.780000,0.745000,0.760000,0.650000,0.755000,0.775000,0.675000,0.820000,0.555000,0.895000,0.930000,0.870000,0.740000,0.780000,0.730000,0.775000,0.750000,0.835000,0.895000,0.715000,0.740000,0.625000,0.795000,0.745000,0.580000];
TN_ratio = 1 - TP_ratio;

org_dir = pwd;
srcF = 'results_bruno';
base_txt = 'FC_acc+thresh_ARTICLE_NEW_TEST_'

sz_layers = 17; 
sz_tst_objs = 1;
sz_in_tst_objs = 59;
nb_eps = 20;

nb_train = 1;
acc_2_lyr = zeros(sz_tst_objs,sz_in_tst_objs,3,sz_layers,nb_train,nb_eps,12);

best_2_lyr = zeros(sz_tst_objs,sz_in_tst_objs,3,sz_layers,12);
worst_2_lyr = zeros(sz_tst_objs,sz_in_tst_objs,3,sz_layers,12);

x_axis_print = linspace(1,sz_layers,sz_layers);

cd(srcF);
pause('on');

for cnt_hid = 2
	max_cnt_train = 1;

	for cnt_layer = 1:sz_layers
		for cnt_train = 1:max_cnt_train
			for cnt_tst_gen = 1:1
				for cnt_tst_obj_nb = 1:test_objs_blocks

					cnt_tst_obj = cnt_tst_obj_nb;

					f_str = [base_txt, char(layer_name(cnt_layer)), '_epoch_', num2str(nb_eps), '_hidden_lyr_', num2str(cnt_hid), '_train_nb_', num2str(cnt_train), '_', 'video', num2str(cnt_tst_obj_nb-1), '.csv'];

					disp(f_str);
					tmp_read = csvread(f_str,1,0);

					tmp_read_f = tmp_read(1:3:60,:);
					tmp_read_b = tmp_read(2:3:60,:);
					tmp_read_bh = tmp_read(3:3:60,:);

					if (cnt_hid == 2)
						acc_2_lyr(cnt_tst_gen,cnt_tst_obj,1,cnt_layer,cnt_train,:,:) = tmp_read_f;
						acc_2_lyr(cnt_tst_gen,cnt_tst_obj,2,cnt_layer,cnt_train,:,:) = tmp_read_b;
						acc_2_lyr(cnt_tst_gen,cnt_tst_obj,3,cnt_layer,cnt_train,:,:) = tmp_read_bh;
					end;

				end;
			end;
		end;
	end;
end;
cd(org_dir);


tic;

for cnt_hid = 2
	for cnt_tst_gen = 1:1
		for cnt_tst_obj_nb = 1:test_objs_blocks
			cnt_tst_obj = cnt_tst_obj_nb;
			for cnt_layer = 1:sz_layers
				for cnt_tmp = 1:3

					nb_train = 1;
				
					best_acc = 0;
					best_TP = 0;
					best_TN = 0;
					best_FP = 0;
					best_FN = 0;
					best_epc = 0;
					best_thresh = 0;
					best_TNN = 0;
					best_MNN = 0;
					best_elap = 0;
					best_elap_d = 0;
					best_DIS = 999;
					
					for cnt_train = 1:nb_train
						for cnt_eps = 1:nb_eps
							curr_acc = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,4);
							if(best_DIS > curr_acc)
								best_acc = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,3);
								best_DIS = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,4);
								best_TP = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,5);
								best_TN = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,6);
								best_FP = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,7);
								best_FN = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,8);
								best_epc = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,1);
								best_thresh = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,2);
								best_TNN = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,9);
								best_MNN = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,10);
								best_elap = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,11);
								best_elap_d = acc_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,cnt_train,cnt_eps,12);
								
								if (best_epc == 0)
									disp('FOUND 2 LYR');
								end;
								
							end;
						end; 
						
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,1) = best_epc;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,2) = best_thresh;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,3) = best_acc;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,4) = best_DIS;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,5) = best_TP;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,6) = best_TN;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,7) = best_FP;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,8) = best_FN;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,9) = best_TNN;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,10) = best_MNN;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,11) = best_elap;
						best_2_lyr(cnt_tst_gen,cnt_tst_obj,cnt_tmp,cnt_layer,12) = best_elap_d;
					
						
					end;
				
				end;
			end;
		end;
	end;
end;


toc;	 
	





Layer = reshape(x_axis_print,[17,1]);
xls_str = strcat('FC_', 'video_ALL', '.csv');

Acc_50 = zeros(17,1);
Acc_best = zeros(17,1);
Acc_bestH = zeros(17,1);
DIS_50 = zeros(17,1);
DIS_best = zeros(17,1);
DIS_bestH = zeros(17,1);
TP_50 = zeros(17,1);
TP_best = zeros(17,1);
TP_bestH = zeros(17,1);
TN_50 = zeros(17,1);
TN_best = zeros(17,1);
TN_bestH = zeros(17,1);
FP_50 = zeros(17,1);
FP_best = zeros(17,1);
FP_bestH = zeros(17,1);
FN_50 = zeros(17,1);
FN_best = zeros(17,1);
FN_bestH = zeros(17,1);
Epc_50 = zeros(17,1);
Epc_best = zeros(17,1);
Epc_bestH = zeros(17,1);
Thresh = zeros(17,1);
Thresh_H = zeros(17,1);
Top_NN = zeros(17,1);
Mid_NN = zeros(17,1);
elap = zeros(17,1);

TP_sum_ratio = 0;
FP_sum_ratio = 0;
TN_sum_ratio = 0;
FN_sum_ratio = 0;

for cnt_tst_gen = 1:1
	for cnt_tst_obj_nb = 1:test_objs_blocks
		cnt_tst_obj = cnt_tst_obj_nb;

		%BEST_2_LYR
		Acc_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,3)/100;
		Acc_50_tmp = reshape(Acc_50_tmp, [17,1]);
		
		Acc_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,3)/100;
		Acc_best_tmp = reshape(Acc_best_tmp, [17,1]);
		
		Acc_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,3)/100;
		Acc_bestH_tmp = reshape(Acc_bestH_tmp, [17,1]);
		
		TP_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,5)/100;
		TP_50_tmp = reshape(TP_50_tmp, [17,1]);
		
		TP_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,5)/100;
		TP_best_tmp = reshape(TP_best_tmp, [17,1]);
		
		TP_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,5)/100;
		TP_bestH_tmp = reshape(TP_bestH_tmp, [17,1]);
		
		TN_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,6)/100;
		TN_50_tmp = reshape(TN_50_tmp, [17,1]);
		
		TN_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,6)/100;
		TN_best_tmp = reshape(TN_best_tmp, [17,1]);
		
		TN_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,6)/100;
		TN_bestH_tmp = reshape(TN_bestH_tmp, [17,1]);
		
		FP_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,7)/100;
		FP_50_tmp = reshape(FP_50_tmp, [17,1]);
		
		FP_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,7)/100;
		FP_best_tmp = reshape(FP_best_tmp, [17,1]);
		
		FP_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,7)/100;
		FP_bestH_tmp = reshape(FP_bestH_tmp, [17,1]);
		
		FN_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,8)/100;
		FN_50_tmp = reshape(FN_50_tmp, [17,1]);
		
		FN_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,8)/100;
		FN_best_tmp = reshape(FN_best_tmp, [17,1]);
		
		FN_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,8)/100;
		FN_bestH_tmp = reshape(FN_bestH_tmp, [17,1]);
		
		Epc_50_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,1,:,1);
		Epc_50_tmp = reshape(Epc_50_tmp, [17,1]);
		
		Epc_best_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,1);
		Epc_best_tmp = reshape(Epc_best_tmp, [17,1]);
		
		Epc_bestH_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,1);
		Epc_bestH_tmp = reshape(Epc_bestH_tmp, [17,1]);
		
		Thresh_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,2)/100;
		Thresh_tmp = reshape(Thresh_tmp, [17,1]);
		
		Thresh_H_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,2)/100;
		Thresh_H_tmp = reshape(Thresh_H_tmp, [17,1]);
		
		Top_NN_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,9);
		Top_NN_tmp = reshape(Top_NN_tmp,[17,1]);
		
		Mid_NN_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,2,:,10);
		Mid_NN_tmp = reshape(Mid_NN_tmp,[17,1]);
		
		elap_tmp = best_2_lyr(cnt_tst_gen,cnt_tst_obj,3,:,11);
		elap_tmp = reshape(elap_tmp, [17,1]);
		
		TP_50_tmp = TP_50_tmp * TP_ratio(cnt_tst_obj);
		TP_best_tmp = TP_best_tmp * TP_ratio(cnt_tst_obj);
		TP_bestH_tmp = TP_bestH_tmp * TP_ratio(cnt_tst_obj);
		TP_sum_ratio = TP_sum_ratio + TP_ratio(cnt_tst_obj);
		
		TN_50_tmp = TN_50_tmp * TN_ratio(cnt_tst_obj);
		TN_best_tmp = TN_best_tmp * TN_ratio(cnt_tst_obj);
		TN_bestH_tmp = TN_bestH_tmp * TN_ratio(cnt_tst_obj);
		TN_sum_ratio = TN_sum_ratio + TN_ratio(cnt_tst_obj);
		
		FN_50_tmp = FN_50_tmp * TP_ratio(cnt_tst_obj);
		FN_best_tmp = FN_best_tmp * TP_ratio(cnt_tst_obj);
		FN_bestH_tmp = FN_bestH_tmp * TP_ratio(cnt_tst_obj);
		FN_sum_ratio = FN_sum_ratio + TP_ratio(cnt_tst_obj);
		
		FP_50_tmp = FP_50_tmp * TN_ratio(cnt_tst_obj);
		FP_best_tmp = FP_best_tmp * TN_ratio(cnt_tst_obj);
		FP_bestH_tmp = FP_bestH_tmp * TN_ratio(cnt_tst_obj);
		FP_sum_ratio = FP_sum_ratio + TN_ratio(cnt_tst_obj);
		
		Acc_50_tmp = TP_50_tmp + TN_50_tmp;
		Acc_best_tmp = TP_best_tmp + TN_best_tmp;
		Acc_bestH_tmp = TP_bestH_tmp + TN_bestH_tmp;
		
		Acc_50 = Acc_50 + Acc_50_tmp;
		Acc_best = Acc_best + Acc_best_tmp;
		Acc_bestH = Acc_bestH + Acc_bestH_tmp; 
		
		TP_50 = TP_50 + TP_50_tmp;
		TP_best = TP_best + TP_best_tmp;
		TP_bestH = TP_bestH + TP_bestH_tmp;
		
		TN_50 = TN_50 + TN_50_tmp;
		TN_best = TN_best + TN_best_tmp;
		TN_bestH = TN_bestH + TN_bestH_tmp;
		
		FP_50 = FP_50 + FP_50_tmp;
		FP_best = FP_best + FP_best_tmp;
		FP_bestH = FP_bestH + FP_bestH_tmp;
		
		FN_50 = FN_50 + FN_50_tmp;
		FN_best = FN_best + FN_best_tmp;
		FN_bestH = FN_bestH + FN_bestH_tmp;
		
		Epc_50 = Epc_50 + Epc_50_tmp;
		Epc_best = Epc_best + Epc_best_tmp;
		Epc_bestH = Epc_bestH + Epc_bestH_tmp;
		
		Thresh = Thresh + Thresh_tmp;
		Thresh_H = Thresh_H + Thresh_H_tmp;
		
		Top_NN = Top_NN + Top_NN_tmp;
		Mid_NN = Mid_NN + Mid_NN_tmp;
		
		elap = elap + elap_tmp;
		
	end;
end;

TOTAL_VIDS = 59;



TP_50 = TP_50 / TP_sum_ratio;
TP_best = TP_best / TP_sum_ratio;
TP_bestH = TP_bestH / TP_sum_ratio;

FN_50 = FN_50 / FN_sum_ratio;
FN_best = FN_best / FN_sum_ratio; 
FN_bestH = FN_bestH / FN_sum_ratio;

TN_50 = TN_50 / TN_sum_ratio;
TN_best = TN_best / TN_sum_ratio;
TN_bestH = TN_bestH / TN_sum_ratio; 

FP_50 = FP_50 / FP_sum_ratio;
FP_best = FP_best / FP_sum_ratio;
FP_bestH = FP_bestH / FP_sum_ratio;

Acc_50 = Acc_50 / TOTAL_VIDS;
Acc_best = Acc_best / TOTAL_VIDS;
Acc_bestH = Acc_bestH / TOTAL_VIDS;

Epc_50 = Epc_50 / TOTAL_VIDS;
Epc_best = Epc_best / TOTAL_VIDS;
Epc_bestH = Epc_bestH / TOTAL_VIDS;

Thresh = Thresh / TOTAL_VIDS;
Thresh_H = Thresh_H / TOTAL_VIDS;

Top_NN = Top_NN / TOTAL_VIDS;
Mid_NN = Mid_NN / TOTAL_VIDS; 
elap = elap / TOTAL_VIDS;

for a = 1:17
	DIS_50(a) = ((1 - TP_50(a))^2 + (FP_50(a))^2 )^(1/2);
	DIS_best(a) = ((1 - TP_best(a))^2 + (FP_best(a))^2 )^(1/2);
	DIS_bestH(a) = ((1 - TP_bestH(a))^2 + (FP_bestH(a))^2 )^(1/2);
end;

 
table_wr = table(Layer, Acc_50, DIS_50, TP_50, TN_50, FP_50, FN_50, Epc_50, Acc_best, DIS_best, TP_best, TN_best, FP_best, FN_best, Epc_best, Thresh, Acc_bestH, DIS_bestH, TP_bestH, TN_bestH, FP_bestH, FN_bestH, Epc_bestH, Thresh_H, Top_NN, Mid_NN, elap);
writetable(table_wr,xls_str);

