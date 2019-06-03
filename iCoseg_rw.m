clc;clear all;

for qqq=[3:40] 
%%initialization 
beta = 90;
addpath('.\others\');
rootresult=strcat('./result_iCoseg/');
namepath= '.\images_iCoseg\';
foldername=dir(namepath);
namepath_salmat= '.\images_iCoseg_salmat\';
foldername_salmat=dir(namepath_salmat);
ObjPath = '.\iCoSeg_UFO_objectness\';   
file_path_superpixel_map='.\superpixel_map\';
file_path_superpixel_column_file='.\superpixel_column_file\';
file_path_first_salmap='.\first_step_sal_output\';
S_sal_final=zeros(300,1);
[m n]=size(foldername);
p=3;
disp(qqq);
file_path=strcat(namepath,foldername(qqq,1).name, '/');
files=dir([file_path '*.jpg']);
result=strcat(rootresult,foldername(qqq,1).name);
mkdir(result);

Img_num=size(files,1);
ScaleH=128; 
ScaleW=128;
Bin_num=min(max(2*Img_num,10),30);
Bin_num=50;
%% ------ Obtain the co-saliency for multiple images-------------
%----- obtaining the features -----
daxiao=zeros(2,Img_num);

for i=1:Img_num
   disp(i)
   path=strcat(file_path, files(i,1).name); %name of image
   [p,q,t]=size(imread(path)); 
   daxiao(1,i)=p;
   daxiao(2,i)=q;
   [imvector img DisVector]=GetImVector(path, ScaleH, ScaleW,0); % imvector: colourspace
    if i==1
        All_vector=imvector;
        All_img=img;
    else 
        All_vector=[All_vector; imvector];
        All_img=[All_img img];
    end
end
[idx, ctrs, bCon, sumD, D] = litekmeans(All_vector, Bin_num,'MaxIter',5,'Replicates',1);
for i=1:Img_num
    %%loads prev. saliency map from peng et.al. and stacks all images
    path=strcat(file_path, files(i,1).name);
   disp(i);
path_superpixel_column=strcat(file_path_superpixel_column_file, strcat(files(i,1).name(1:end-4)), '.mat');
   load (path_superpixel_column);
   if i==1
        S_sal_vector=S_sal;
    else 
        S_sal_vector=[S_sal_vector S_sal];
   end
    imgVals1=All_img;
    
end
%%stack from all_img 3D to imagvals 2D
    tmp = imgVals1(:,:,1);
    imgVals = tmp(:);
    tmp = imgVals1(:,:,2);
    imgVals(:,2) = tmp(:);
    tmp = imgVals1(:,:,3);
    imgVals(:,3) = tmp(:);
    gt_path='.\saliencymap\';
path_salName2='.\results\';
 for ii=1:Img_num
   path=strcat(gt_path, strrep(files(ii,1).name,'.jpg','.png')); %edited
   f=im2single(imread(path)); %  converts to single
   f=imresize(f,[ScaleH ScaleW]); %resize by interpolation & antialiasing
   th=mean(f(:)); %of one single image
   f=f>=(th); % if greater than th then 1 else 0- Binarize by a threshold -this is the query vector
       for j=1:ScaleH
           for i=1:ScaleW
               if f(j, i)>=0 %greater than threshold- is the queries -F/G
                    Yt1(j +(i-1)*ScaleH+ScaleH*ScaleW*(mod(ii-1,Img_num)))=f(j, i); %%?
               end
           end
       end
end   

th1= (mean(S_sal_vector) + max(S_sal_vector)) / 2 ; 
th2 = mean(S_sal_vector);
alpha = 0.99; 
mu = (1-alpha) / alpha;
for i=1:Img_num
     path_superpixel_map=strcat(file_path_superpixel_map, strcat(files(i,1).name(1:end-4)), '.mat');
   load (path_superpixel_map);
   %%load superpixel column matrix
   path_superpixel_column=strcat(file_path_superpixel_column_file, strcat(files(i,1).name(1:end-4)), '.mat');
   load (path_superpixel_column);sz=size(S_sal);
   [seedAll, label] = seed4RW(S_sal', th1, th2);
   if i==1
    for ih = 1:length(seedAll)
        [seedY, seedX] = find(superpixel == seedAll(ih));
        seedXM = round(mean(seedX));
        seedYM = round(mean(seedY));
        seedAll(ih) = (seedXM - 1) * 128 + seedYM;
    end
    seedAllLabel = sortrows([seedAll,label]);
    diffSeedAllLabel = diff(seedAllLabel(:,1));
    seedAllLabel(diffSeedAllLabel == 0, :) = [];
    seedAll = seedAllLabel(:,1);
    label = seedAllLabel(:,2);   
    
   else
        for ih = 1:length(seedAll)
        [seedY, seedX] = find(superpixel == seedAll(ih));
        seedY=seedY+128*(i-1);
        seedXM = round(mean(seedX));
        seedYM = round(mean(seedY));
        seedAll(ih) = (seedXM - 1) * 128 + seedYM; 
    end
    seedAllLabel = sortrows([seedAll,label]);
    diffSeedAllLabel = diff(seedAllLabel(:,1));
    seedAllLabel(diffSeedAllLabel == 0, :) = [];
    seedAll = seedAllLabel(:,1);
    label = seedAllLabel(:,2); 
   end
    
    
     if i==1
       seedAll_final=seedAll;
        label_final=label;
     else 
        seedAll_final=[seedAll_final ;seedAll];
        label_final=[label_final; label];
   end

end
seedAll=seedAll_final;label=label_final;  
    %Build graph
    n1=128;m1=128*Img_num; %*Img_num
    N=m1*n1;
    edges=[(1:N)',((1:N)+1)'];
    edges=[[edges(:,1);(1:N)'],[edges(:,2);(1:N)'+m1]];
    excluded=find((edges(:,1)>N)|(edges(:,1)<1)|(edges(:,2)>N)| ...
        (edges(:,2)<1));
    edges([excluded;(m1:m1:((n1-1)*m1))'],:)=[]; 

    dis = sqrt(sum((imgVals(edges(:,1),:) - imgVals(edges(:,2),:)).^2, 2));
    dis = normalize(dis);
    weights = exp(-beta*dis); %%add more features
    N=max(max(edges));
    W=sparse([edges(:,1);edges(:,2)],[edges(:,2);edges(:,1)],[weights;weights],N,N);
    D = diag(sum(W));
    L=D-W;
    
    labelAdjust = min(label); 
    label = label - labelAdjust + 1; %Adjust labels to be > 0
    labelRecord(label) = 1;
    labelPresent = find(labelRecord);
    labelNum = length(labelPresent);
     %Set up Dirichlet problem
    bound = zeros(length(seedAll), labelNum);
    for k=1:labelNum
        bound(:,k) = (label(:) == labelPresent(k)); %2 class construction
    end
    %Solve the combinatorial Dirichlet problem
    saliencyFull = zeros(128,128);
   % saliencyFull1 = zeros(m1, n1);
    for ij=1: Img_num
        path_superpixel_map=strcat(file_path_superpixel_map, strcat(files(ij,1).name(1:end-4)), '.mat');
        load (path_superpixel_map);
        path_superpixel_column=strcat(file_path_superpixel_column_file, strcat(files(ij,1).name(1:end-4)), '.mat');
        load (path_superpixel_column);
    for ijj = 1:length(S_sal)
        saliencyFull(superpixel == ijj) = S_sal(ijj);
    end
    if ij==1
        saliencyFull1=saliencyFull;
    else 
        saliencyFull1=[saliencyFull1, saliencyFull]; %%??
    end
    end
    
    y = saliencyFull1(:); %load all_vector 
    y = (y - min(y)) / (max(y) - min(y)); %%correct
    y = [y, (1-y)];
    index = seedAll(:);
    N = length(L);
    antiIndex = 1:N;
    antiIndex(index) = [];%
    antiY = y; %norm sal values
    antiY(index,:) = [];
    b = -L(antiIndex,index)*(bound);
    muI = sparse((1:length(antiIndex)), (1:length(antiIndex)), mu*ones(length(antiIndex),1));
    x = (L(antiIndex,antiIndex) + muI)\(mu * antiY + b);
    probabilities = zeros(size(bound));
    probabilities(index,:) = bound;
    probabilities(antiIndex,:) = x;
    probabilities = reshape(probabilities,[m1 n1 labelNum]); 
    wid=[m1,n1,1,m1,1,n1];
    sal = probabilities(:,:,1);
    salOutput = zeros(wid(1),wid(2));
	salOutput(wid(3):wid(4),wid(5):wid(6)) = sal; 
    for ik=1:Img_num
       score_ini1=salOutput(((ik-1)*ScaleH+1:ik*ScaleH), :);
       score_ini1=score_ini1(:);
       if ik==1
           score_ini=score_ini1;
       else
           score_ini=[score_ini;score_ini1];
       end
    end
    %%saliency map saving
    for i=1:Img_num
    salOutput1=salOutput((i-1)*128+1:i*128,:); 
  path_salName1= strcat(path_salName2, strcat(files(i,1).name(1:end-4)), '_rcrr.png');
   imwrite(salOutput1, [path_salName1,strcat(files(i,1).name(1:end-4)), '_rrwr.png']); 
     [L4,N4] = superpixels(salOutput1,250); 
       path_salName=strcat(file_path_first_salmap, strcat(files(i,1).name(1:end-4)), '_rcrr.mat');
     save(path_salName,'L4');
      clear L1
       for iu = 1:N4
       L1(iu,1)= mean(salOutput1(L4 == iu)) ;
       end
    path_salName=strcat(file_path_first_salmap, strcat(files(i,1).name(1:end-4)), '.mat');
     save(path_salName,'L1');
      
    end

gt_path='.\saliencymap\';
Sal_co=zeros(Bin_num,Img_num);

for i=1:Img_num
    index_vec = idx((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW);
    index_sal = Yt1((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW);
    Sal_sum=zeros(Bin_num,1);
    Sal_count=ones(Bin_num,1);
    for j=1:ScaleH*ScaleW
        Sal_sum(index_vec(j))=Sal_sum(index_vec(j))+index_sal(j);
        Sal_count(index_vec(j))=Sal_count(index_vec(j))+1;
    end
    Sal_co(:,i)=Sal_sum(:)./Sal_count(:);
end
Sal_distribute=sum(Sal_co,2)./Img_num;
Sal_distribute=Sal_distribute>=0.6526;
for ii=1:Img_num
    path_salName=strcat(file_path_first_salmap, strcat(files(ii,1).name(1:end-4)), '.mat');
   load (path_salName); 
   S_sal_2=L1;
       path_salName=strcat(file_path_first_salmap, strcat(files(ii,1).name(1:end-4)), '_rcrr.mat');
   load( path_salName); 
   superpixel_L=L4;
   th11= (mean(S_sal_2) + max(S_sal_2)) / 2 ;
th21 = mean(S_sal_2);
alpha = 0.99; 
mu = (1-alpha) / alpha;
[seedAll, label] = seed4RW(S_sal_2, th11,th21);
   for i = 1:length(seedAll)
        [seedY, seedX] = find(superpixel_L == seedAll(i));
        seedXM = round(mean(seedX));
        seedYM = round(mean(seedY));
        seedAll(i) = (seedXM - 1) * 128 + seedYM;
    end
    seedAllLabel = sortrows([seedAll,label]);
    diffSeedAllLabel = diff(seedAllLabel(:,1));
    seedAllLabel(diffSeedAllLabel == 0, :) = [];
    seedAll = seedAllLabel(:,1);
    label = seedAllLabel(:,2);     
    n1=128;m1=128*Img_num;
    N=m1*n1;
    edges=[(1:N)',((1:N)+1)'];
    edges=[[edges(:,1);(1:N)'],[edges(:,2);(1:N)'+m1]];
    excluded=find((edges(:,1)>N)|(edges(:,1)<1)|(edges(:,2)>N)| ...
        (edges(:,2)<1));
    edges([excluded;(m1:m1:((n1-1)*m1))'],:)=[]; 

    dis = sqrt(sum((imgVals(edges(:,1),:) - imgVals(edges(:,2),:)).^2, 2));
    dis = normalize(dis);
    weights = exp(-beta*dis);
    N=max(max(edges));
    W=sparse([edges(:,1);edges(:,2)],[edges(:,2);edges(:,1)],[weights;weights],N,N);
    D = diag(sum(W));
    L=D-W;
    
    labelAdjust = min(label); 
    label = label - labelAdjust + 1; 
    labelRecord(label) = 1;
    labelPresent = find(labelRecord);
    labelNum = length(labelPresent);
    
    
    bound = zeros(length(seedAll), labelNum);
    for k=1:labelNum
        bound(:,k) = (label(:) == labelPresent(k)); %2 class construction
    end
     %Solve the combinatorial Dirichlet problem
    saliencyFull = zeros(size(superpixel_L));
    for i = 1:length(S_sal_2)
        saliencyFull(superpixel_L == i) = S_sal_2(i);
    end
    
        for ij=1: Img_num
          path_salName=strcat(file_path_first_salmap, strcat(files(ij,1).name(1:end-4)), '.mat');
          load (path_salName); 
          S_sal_2=L1;
          path_salName=strcat(file_path_first_salmap, strcat(files(ij,1).name(1:end-4)), '_rcrr.mat');
          load( path_salName); 
          superpixel_L=L4;
    for ijj = 1:length(S_sal_2)
        saliencyFull(superpixel_L == ijj) = S_sal_2(ijj);
    end
    if ij==1
        saliencyFull1=saliencyFull;
    else 
        saliencyFull1=[saliencyFull1, saliencyFull]; 
    end
        end
    
    
    y = saliencyFull1(:); %load all_vector
    y = (y - min(y)) / (max(y) - min(y));
    y = [y, (1-y)];
    index = seedAll(:);
    N = length(L);
    antiIndex = 1:N;
    antiIndex(index) = [];
    antiY = y; %norm sal values
    antiY(index,:) = [];
    b = -L(antiIndex,index)*(bound);
    muI = sparse((1:length(antiIndex)), (1:length(antiIndex)), mu*ones(length(antiIndex),1));
    x = (L(antiIndex,antiIndex) + muI)\(mu * antiY + b);
    probabilities = zeros(size(bound));
    probabilities(index,:) = bound;
    probabilities(antiIndex,:) = x;
    probabilities = reshape(probabilities,[m1 n1 labelNum]);
    wid=[m1,n1,1,m1,1,n1];
    sal = probabilities(:,:,1);
    salOutput = zeros(wid(1),wid(2));
	salOutput(wid(3):wid(4),wid(5):wid(6)) = sal; %second step sal map
    if ii==1
           salOutput2=salOutput;
       else
           salOutput2=[salOutput2;salOutput];
       end
   end

 for ikk=1:Img_num
    for ik=1:Img_num 
       score_ini2=salOutput2(((ik-1)*ScaleH+1:ik*ScaleH), :);
       score_ini2=score_ini2(:);
       if ik==1
           score3=score_ini2;
       else
           score3=[score3;score_ini2];
       end
       
    end
    score4(:,ikk) =score3;
 end

%% first
   score=Co_saliency(score4,Img_num,ScaleH,ScaleW,Sal_distribute,idx);
%%
     for i=1:Img_num
        path=strcat(file_path, files(i,1).name);
        vec=score((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW);
        vec=Nor(vec);
        SM1=reshape(vec,[ScaleH ScaleW]);
        SM1=imresize(SM1,[daxiao(1,i) daxiao(2,i)]);
        ObPath=strcat(ObjPath, strcat(files(i,1).name(1:end-4)), '.mat');
        load(ObPath);
        SM1=SM1  
        SM=MeanShforSP(path,SM1);
        imwrite(SM,strcat(result,'\', strrep(files(i,1).name,'.jpg','_co.png')));  
     end
     clear all;
clear model;
toc;
end




