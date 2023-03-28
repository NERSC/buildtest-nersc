from lmod.moduleloadtest import ModuleLoadTest
test = ModuleLoadTest(debug=True, login=True, name=['e4s'])
results = test.get_results()
print(results)
assert(results['failed'] == 0)
