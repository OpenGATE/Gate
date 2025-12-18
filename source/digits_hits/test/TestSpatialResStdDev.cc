/*
  Simple unit test to verify GateDigi stores spatial-resolution stddevs
  This executable is optional and requires a Geant4-enabled build.
*/
#include "GateDigi.hh"
#include <iostream>
#include <cmath>

int main()
{
    GateDigi digi;

    const double x = 1.2345;
    const double y = 2.3456;
    const double z = 3.4567;

    digi.SetSpatialRes2DStdDevX(x);
    digi.SetSpatialRes2DStdDevY(y);
    digi.SetSpatialRes2DStdDevZ(z);

    const double eps = 1e-12;

    if (std::fabs(digi.GetSpatialRes2DStdDevX() - x) > eps) {
        std::cerr << "Get/Set mismatch for StdDevX: got " << digi.GetSpatialRes2DStdDevX() << " expected " << x << "\n";
        return 1;
    }
    if (std::fabs(digi.GetSpatialRes2DStdDevY() - y) > eps) {
        std::cerr << "Get/Set mismatch for StdDevY: got " << digi.GetSpatialRes2DStdDevY() << " expected " << y << "\n";
        return 2;
    }
    if (std::fabs(digi.GetSpatialRes2DStdDevZ() - z) > eps) {
        std::cerr << "Get/Set mismatch for StdDevZ: got " << digi.GetSpatialRes2DStdDevZ() << " expected " << z << "\n";
        return 3;
    }

    std::cout << "TestSpatialResStdDev: OK\n";
    return 0;
}
