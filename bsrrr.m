function[C_post,C_mean,C_median,r_hat,C_RR]=bsrrr(Y,X,BURNIN,MCMC,thin,r,type)

%% Function to implement simultaneous dimension reduction and variable selection in multiple response linear model from Chakraborty, Bhattacharya, Mallick (2016)%%
%% Model: Y=XC+E, where Y is a n*q mtrices with each row containing information on q measurements on n subjects, X is a n*p matrix of potential predictors %%
%% C is p*q matrix of regression coefficients with possibly low rank and many zero rows. E is a p*q matrix of unobservable errors.%%
%% Prior: Write C=BA' where B is p*q and A is q*q. Let b_{jh} be the element of the j-th row and h-th column. Similarly define a_{hk}. We assume %%
%% b_{jh} \sim N(0,\lambda_{jh}^2\tau_h^2): independent Horseshoe (Carvalho et al. (2008)) priors on each column of B and independent N(0,1) priors on the elements of A.%%
%% Letting E=(e_1',...,e_n')' we assume e_i \sim N(0,\Sigma) where \Sigma=diag(\sigma_1^2,...,\sigma_q^2), idiosyncratic error variances for each response. Apriori we%%
%% assume \sigma_h^2 \propto 1/\sigma_h^2.%%

%% Input: Y= response matrix, size n*q %%
%%        X= design matrix, size n*p %%
%%        BURNIN= number of BURNIN samples to throw away for posterior calculations, default is 200 %%
%%        MCMC= number of MCMC samples to keep, default is 1000 %%
%%        thin= thinning parameter of the MCMC chain, default is 1 %%
%%        r= initial guess of the rank of C %%
%%        method= logical indicator used for whether dimension reduction and variable selection is required. Options are 'dim_red', 'var_sel', 'both'. Default is 'both'.

%% Output: C_post= posterior samples of the matrix C, a 3 dimensional array of size MCMC*p*q. %%
%%         C_mean= entrywise mean of C_post, a p*q matrix %%
%%         C_median= entrywise mean of C_post, a p*q matrix %%
%%         r_hat= estimated rank of C %%
%%         C_RR= row and/or rank selected estimate of C %% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic;
if nargin==5
   r=size(Y,2);
end
   
nrun=BURNIN+MCMC;
effsamp=(nrun-BURNIN)/thin;
[n,p]=size(X);
q=size(Y,2);

%%%% MCMC initial values  %%%%%
B=zeros(p,r);
A=normrnd(0,1,[q,r]);
sigmasq=ones(q,1);
Lambda=ones(p,r);
tau=ones(r,1);
y=reshape(Y',[],1);

%%%%%%%% MCMC storage %%%%%%%%%

C_store=zeros(MCMC,p,q);
pred_store=zeros(MCMC,n,q);
Lambda_store=zeros(MCMC,p,r);
tau_store=zeros(MCMC,r);
sigmasq_store=zeros(MCMC,q);
%r1=zeros(MCMC,1);
%r2=zeros(MCMC,1);

%% Start MCMC %%
for i=1:nrun

  if mod(i,100) == 0
        disp(i)
  end
   
   %%%% Update B using vectorization  %%%%

  X_tilde=kron(X,A);
  X_tilde=bsxfun(@times,X_tilde,repmat(sqrt(1./sigmasq),[n,1]));
  omega=reshape((Lambda.^2)',[],1).* repmat(tau.^2,[p,1]);
  y_star=repmat(sqrt(1./sigmasq),[n,1]).*y;

  
  %%% Use algo from Bhattacharya et al. %%%
  u=normrnd(zeros(p*r,1),sqrt(omega),p*r,1);
  delta=normrnd(0,1,n*q,1);
  v=X_tilde*u+delta;
  w=(X_tilde*diag(omega)*X_tilde'+eye(n*q))\(y_star-v);
  beta=u+diag(omega)*X_tilde'*w;
  B=(reshape(beta,r,p))';
      
  %%%% Update A in a block %%%%
  X_star=kron(X*B,eye(q));
  X_star=bsxfun(@times,X_star,repmat(sqrt(1./sigmasq),[n,1]));

  %%% Use algo from Rue (2001) %%%
  L=chol(X_star'*X_star+eye(r*q),'lower');
  v=L\X_star'*y_star;
  m=L'\v;
  w=L'\randn(r*q,1);
  a=m+w;
  A=reshape(a,[q,r]);

  %% Update Lambda %%
  %% update lambda_j's in a block using slice sampling %%
  
   for k=1:r
       eta = 1./(Lambda(:,k).^2); 
       upsi = unifrnd(0,1./(1+eta));
       tempps = (B(:,k).^2)/(2*tau(k)^2); 
       ub = (1-upsi)./upsi;
       % now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
       Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
       Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
       up = unifrnd(0,Fub); 
       eta = -log(1-up)./tempps; 
       Lambda(:,k) = 1./sqrt(eta);
   end
      

  %% Update tau %%
  for  k=1:r
       tempt = sum((B(:,k)./Lambda(:,k)).^2)/2; 
       et = 1/tau(k)^2; 
       utau = unifrnd(0,1/(1+et));
       ubt = (1-utau)/utau; 
       Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
       Fubt = max(Fubt,1e-8); % for numerical stability
       ut = unifrnd(0,Fubt); 
       et = gaminv(ut,(p+1)/2,1/tempt); 
       tau(k) = 1/sqrt(et);
  end

  %% Update sigmasq %%
  M=X*B*A';
  for k=1:q
      S_k=(Y(:,k)-M(:,k))'*(Y(:,k)-M(:,k));
      sigmasq(k,1)=1/gamrnd(n/2,1/(0.5*S_k),1);
  end
  C=B*A';
  %MSE_test=sum(sum((C-C_T).^2))/p/q;
  %disp(MSE_test)
  if i > BURNIN && mod(i, thin)== 0
     %d=svd(X*C);
     %d1=svd(C);
     %r1((i-BURNIN)/thin,1)=sum(d1>max(sigmasq));
     %d2=svd(Y-X*C);
     %r2((i-BURNIN)/thin,1)=sum(d>max(d2));
     B_store((i-BURNIN)/thin,:,:) = B;
     A_store((i-BURNIN)/thin,:,:) = A;
     C_store((i-BURNIN)/thin,:,:) = C;
     pred_store((i-BURNIN)/thin,:,:) = X*C;
     Lambda_store((i-BURNIN)/thin,:,:) = Lambda;
     tau_store((i-BURNIN)/thin,:)= tau;
     sigmasq_store((i-BURNIN)/thin,:)=sigmasq';
     
  end
end
%t=toc;
sigmasq_mean=mean(sigmasq_store,1);
hat_sigma=diag(sigmasq_mean);
C_mean=reshape(mean(C_store,1),p,q);
C_median=reshape(median(C_store,1),p,q);
C_post=C_store;
%mse_mean=norm(C_mean-C_T,'fro')/p/q;
%mse_median=norm(C_median-C_T,'fro')/p/q;
%mspe_mean=norm(X*C_mean-X*C_T,'fro')/n/q;
%mspe_median=norm(X*C_median-X*C_T,'fro')/n/q;
%mspe_median_test=norm((X*C_median-X*C_T)*hat_sigma,'fro')/n/q;
%XC_median=reshape(median(pred_store,1),n,q);
%mspe_new=norm(X*C_T-XC_median,'fro')^2/n/q;
%rank_estimate=mean(r2);
switch type
    case 'both'
    [C_R,MSE]=grouplasso(X,C_mean,C_median);
    d=svd(X*C_R);
    d1=svd(Y-X*C_R);
    r_hat=sum(d>max(d1));
    [U,S,V]=svd(C_R);
    C_RR= U(1:p,1:r_hat)*S(1:r_hat,1:r_hat)*V(1:q,1:r_hat)';
    case 'dim_red'
    d=svd(X*C_mean);
    d1=svd(Y-X*C_mean);
    r_hat=sum(d>max(d1));
    [U,S,V]=svd(C_mean);
    C_RR= U(1:p,1:r_hat)*S(1:r_hat,1:r_hat)*V(1:q,1:r_hat)';
    case 'var_sel'
    [C_RR,MSE]=grouplasso(X,C_mean,C_median);
    r_hat=q;
end
