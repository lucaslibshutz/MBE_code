function [Xe,Ye,U,S,th] = calculateEllipseCovMC(X, P, nsig, np) 
    % This functions returns points to draw a 2D ellipse
    %
    %   [Xe,Ye,U,S,th] = calculateEllipseCovMC(X, P, nsig, np)
    % 
    % INPUTS
    %  @param X     2D mean 
    %  @param P     2D covariance matrix 
    %  @param nsig  number of sigmas (default is 1) 
    %  @param np    number of plot points (default is 36) 
    % 
    % OUTPUTS
    %  @param Xe    ellipse x coordinate
    %  @param Ye    ellipse x coordinate
    %  @param U     U from svd of covariance (as a rotation matrix)
    %  @param S     S from svd of covariance
    %  @param th    rotation angle
    %
    % Example/usage:
    %   P=[6.5 2.5;2.5 6.5];
    %   [Xe,Ye,U,S,th] = calculateEllipseCovMC(m, P, 1, 50);
    %   plot(Xe,Ye,'b-');
    %
    %   M Campbell, Jan 2024

    error(nargchk(2, 4, nargin)); 
    if nargin<3, nsig = 1; end 
    if nargin<4, np = 36; end 

    if (length(X) ~= 2) | (length(P) ~= 2), 
        disp('mean and covariance have to be 2D');
        Xe=[];Ye=[];U=[];S=[];th=[];
        return;
    end

    [U,S,V]=svd(P); %svd of the covariance
    %make the U matrix a true rotation matrix: [cos(th) -sin(th);sin(th) cos(th)
    if U(1,1) ~= 0, %just be sure diagonal is not all zeros
        U=U*diag(sign(diag(U)));
    end
    %find rotation angle
    th=-asin(U(1,2))*180/pi;

    s1=sqrt(S(1,1));s2=sqrt(S(2,2));
    x=X(1);
    y=X(2);

    %scale by nsig
    s1=nsig*s1;
    s2=nsig*s2;

    beta = th * (pi / 180); 
    sinbeta = sin(beta); 
    cosbeta = cos(beta); 
 
    alpha = linspace(0, 360, np)' .* (pi / 180); 
    sinalpha = sin(alpha); 
    cosalpha = cos(alpha); 

    Xe = x + (s1 * cosalpha * cosbeta - s2 * sinalpha * sinbeta); 
    Ye = y + (s1 * cosalpha * sinbeta + s2 * sinalpha * cosbeta); 
 
end