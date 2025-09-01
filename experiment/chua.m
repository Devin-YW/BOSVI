% function dy=chua(t, y)
%     dy=zeros(3, 1);
%     alfa=10.0;
%     beta=14.87;
%     a=-1.27;
%     b=-0.65;
%     bp=1.0;
%     dy(1)=alfa*(y(2)-y(1)-(b*y(1)+0.5*(a-b)*(abs(y(1)+bp)-abs(y(1)-bp))));
%     dy(2)=y(1)-y(2)+y(3);
%     dy(3)=-beta*y(2);
% end

function dy=chua(t, y)
%     R1 =22e+3 ; R2 = 220; R3 =22e+3 ; R4  =220 ; R =1600 ;
%     C1 =10e-9 ; C2 =100e-9 ; L =20e-3 ;
%     Ga = -(1/R1) - (1/R4);
%     Gb = -(1/R3) - (1/R4);
%     G = 1/R;
%     alfa = Ga/G;
%     beta = C2/(L*(G.^2));
%     a = C2/C1;
%     b = Gb/G;
    dy=zeros(3, 1);
    alfa=10.0;
    beta=14.87;
    a=-1.27;
    b=-0.75;
    bp=2.0;
    dy(1)=alfa*(y(2)-y(1)-(b*y(1)+0.5*(a-b)*(abs(y(1)+bp)-abs(y(1)-bp))));
    dy(2)=y(1)-y(2)+y(3);
    dy(3)=-beta*y(2);
end