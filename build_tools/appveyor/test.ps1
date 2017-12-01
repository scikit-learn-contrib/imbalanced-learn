# Change to a non-source folder to make sure we run the tests on the
# installed library.
- "cd C:\\"

$installed_imblearn_folder = $(python -c "import os; os.chdir('c:/'); import imblearn;\
print(os.path.dirname(imblearn.__file__))")
echo "imblearn found in: $installed_imblearn_folder"

# --pyargs argument is used to make sure we run the tests on the
# installed package rather than on the local folder
pytest --pyargs imblearn $installed_imblearn_folder
exit $LastExitCode
