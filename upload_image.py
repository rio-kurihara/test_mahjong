from cStringIO import StringI

from boto import connect_s3

# token
access_key = ''
secret_key = ''

# bucket, key
bucket_name = ''
file_key = ''

# S3 からファイルを取得
conn = connect_s3(access_key, secret_key)
bucket = conn.get_bucket(bucket_name)
key = bucket.get_key(file_key)

# key_name で取得できなかった場合 None が返る
if not key:
    return u'ないよー'

# ファイルダウンローして、 StgingIO に書き込み
fp = StringIO()
key.get_contents_to_file(fp)
fp.seek(0)
