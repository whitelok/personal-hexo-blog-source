export BLOG_ROOT_PATH=/Users/karllok/Desktop/whitelok.github.com
hexo clean && hexo gen && cp -r public/* $BLOG_ROOT_PATH && cd $BLOG_ROOT_PATH && git add . --all && git commit -m "update new post" && git push && cd -
