data = xlsread("gpm.xlsx");
x = data(:,2);
y = data(:,3);
z = data(:,1);

xi=linspace(min(x),max(x),30);
yi=linspace(min(y),max(y),30);
[XI, YI]=meshgrid(xi,yi);
ZI = griddata(x,y,z,XI,YI);
colormap('jet');
% for i=1:30
%     for j=1:30
%         if(isnan(ZI(i,j)))
%             ZI(i,j)=0;
%         end
%     end
% end

contourf(XI,YI,ZI);
% colormap('jet');
% scatter(x,y,[],z,'fill');
title('Reflectivity (DBZ) on 2019-06-12 at 23:55:31');
xlabel('Latitudes degress North');
ylabel('Longitudes degrees East');
colorbar;
h = colorbar;
ylabel(h, 'Reflectivity(DBZ)');