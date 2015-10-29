install_auction()
{
	which auction

	if [ "$?" -eq 0 ] # auction is installed
	then
		echo "Install auction"
		pip install --upgrade .
	else
		echo "Upgrade auction"
		pip install .
	fi
}

which pip

if [ "$?" -eq 0 ] # pip is installed
then
	install_auction
else
	echo "Install pip"
	curl -kL https://raw.github.com/pypa/pip/master/contrib/get-pip.py | python
	pip install .
	install_auction
fi
