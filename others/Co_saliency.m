function score=Co_saliency(score4,Img_num,ScaleH,ScaleW,Sal_distribute,idx)
%% normalize score4
for j=1:Img_num %score 4 : first image correspodng more similarity in first column
    score_temp=score4(:,j); %score4 is last  cosal output % get for first image
    score_temp((j-1)*ScaleH*ScaleW+1:j*ScaleH*ScaleW)=0; %set first image (128*128) as 0
    score_temp=Nor(score_temp); % normalize rest
    for i=1:Img_num
        if i==j
            vec=score4((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW,j);
        else
            vec=score_temp((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW);
        end
        score_norm((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW,j)=vec; %311296 *19
    end
end

bool=zeros(ScaleH*ScaleW*Img_num,Img_num);
for j=1:Img_num
    for i=1:Img_num
        vec=score_norm((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW,j);
        th=mean(vec);
        bool((i-1)*ScaleH*ScaleW+1:i*ScaleH*ScaleW,j)=(vec>=th); %threshold for each image in each 311296 set 0/1 accordingly
    end
end
   boo=sum(bool,2);
   boo=boo>fix(Img_num/2); %boo=count , in paper
   
   score_s=zeros(ScaleH*ScaleW*Img_num,1); 
   for i=1:Img_num
       score_s=score_s+score_norm(:,i); %summation part over j
   end
   score_s=score_s./Img_num;
   
   score_m=ones(ScaleH*ScaleW*Img_num,1); 
   for j=1:Img_num
       score_m=score_m.*score_norm(:,j); %multiplication part over j
   end
   
   score=zeros(ScaleH*ScaleW*Img_num,1); 
   for i=1:ScaleH*ScaleW*Img_num
       if boo(i)==1 && Sal_distribute(idx(i))==1   %%idx gives cluster no. sal distribute = on an average has the bin been a query
            score(i)=score_s(i);           %% on an average  the bin has been a query & value above count then sumation
       else
            score(i)=score_m(i);
       end
   end
   score=Gauss_normal(score);

   