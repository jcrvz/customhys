function usage {
       printf "Usage:\n"
       printf " -r                               Remove package's dist folder.\n"
       printf " -u                               Uninstall package.\n"
       printf " -b                               Build package.\n"
       printf " -i                               Install package.\n"
       printf " -p                               Send package to PyPi.\n"
       printf " [no flag]                        Default mode: -crbi (no -p).\n"
       exit 0
}

send_pip() {
  echo "Sending to PyPi..."
  #python -m twine upload --repository testpypi dist/*
  python -m twine upload dist/*
  printf "[Done]\n"
}

remove_package() {
  echo "Removing dist..."
  rm -rf dist
  rm -rf *.egg-info
  printf "[Done]\n"
}

uninstall_package()  {
  echo "Uninstalling..."
  python3 -m pip uninstall customhys
  printf "[Done]\n"
}

build_package() {
  echo "Building package..."
  python3 -m build
  printf "[Done]\n"
}

install_package() {
  echo "Installing package..."
  python3 setup.py sdist
  python3 -m pip install dist/*.tar.gz
  printf "[Done]\n"
}

default () {
  printf "Running with -rcbi...\n\n";
  remove_package;
  uninstall_package;
  build_package;
  install_package;
  exit 1;
  printf "[Done]\n"
}

while getopts 'rubiph' flag; do
  case "${flag}" in
    r) remove_package ;;
    u) uninstall_package ;;
    b) build_package ;;
    i) install_package ;;
    p) send_pip ;;
    h) usage ;;
    *) usage
      exit 1 ;;
  esac
done

if [ $OPTIND -eq 1 ];
then
  default;
fi