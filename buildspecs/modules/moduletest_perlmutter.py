from lmod.moduleloadtest import ModuleLoadTest
test = ModuleLoadTest(debug=True, login=True)
results = test.get_results()
print(results)
assert(results['failed'] == 0)
