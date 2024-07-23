import sys
import os
import toml
from gx_geometry.driver import create_eik_from_vmec, create_eik_from_desc

def run_module():
    
    # read parameters from input file
    input_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        stem = input_file[:-3]
        eiknc = sys.argv[2]
    else:
        stem = input_file[:-3]
        eiknc = stem + ".eik.nc"
    
    params = toml.load(input_file)
    
    geo_option = params.get("Geometry").get("geo_option")
    zeta_center = params.get("Geometry").get("zeta_center", 0.0)
    nz = params.get("Dimensions").get("ntheta")
    npol = params.get("Geometry").get("npol")
    
    geo_file = params.get("Geometry").get("geo_file", None)
    if geo_file is None:
        geo_file = params.get("Geometry").get("vmec_file", None)
        if geo_file is not None:
            geo_option = "vmec"
        else:
            assert False, "Must specify an equilibrium file via 'geo_file' parameter"
    else:
        if geo_file[-3:] == ".nc":
            geo_option = "vmec"
        elif geo_file[-3:] == ".h5":
            geo_option = "desc"
        else:
            assert False, f"geo_file = {geo_file} is of unknown type"

    if geo_option == "vmec":
        # s = psi/psi_LCFS
        try:
            s = params["Geometry"]["torflux"]
        except KeyError:
            s = params["Geometry"]["desired_normalized_toroidal_flux"]
        filename = params.get("Geometry").get("vmec_file", params.get("Geometry").get("geo_file"))
        create_eik_from_vmec(filename, s=s, nz=nz+1, poloidal_turns=npol, theta0=0, zeta0=zeta_center, eik_filename=eiknc, **params)
    elif geo_option == "desc":
        rho = params.get("Geometry").get("rhotor")
        filename = params.get("Geometry").get("geo_file")
        create_eik_from_desc(filename, s=s, nz=nz+1, poloidal_turns=npol, theta0=0, zeta0=zeta_center, eik_filename=eiknc, **params)
    
if __name__ == "__main__":
    run_module()
