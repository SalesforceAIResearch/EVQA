#!/usr/bin/env python3
"""
Test write permissions to Salesforce/ST-Evidence-Instruct repository
"""

from huggingface_hub import HfApi, whoami
import tempfile
import sys

def test_write_permission():
    print("=" * 70)
    print("Testing Salesforce Repository Write Permissions")
    print("=" * 70)

    api = HfApi()
    repo_id = "Salesforce/ST-Evidence-Instruct"

    # Check current user
    print("\n1. Current User Info:")
    try:
        user_info = whoami()
        print(f"   Username: {user_info['name']}")

        # Check organization role
        orgs = user_info.get('orgs', [])
        salesforce_org = None
        for org in orgs:
            if org['name'] == 'Salesforce':
                salesforce_org = org
                break

        if salesforce_org:
            print(f"   ✓ Member of Salesforce organization")
            print(f"   Role in Salesforce: {salesforce_org.get('roleInOrg', 'unknown')}")

            if salesforce_org.get('roleInOrg') == 'write':
                print("   ✅ You have WRITE role in Salesforce org!")
            elif salesforce_org.get('roleInOrg') == 'read':
                print("   ⚠️  You only have READ role in Salesforce org")
            else:
                print(f"   ⚠️  Your role is: {salesforce_org.get('roleInOrg')}")
        else:
            print("   ❌ Not a member of Salesforce organization")
            return False

    except Exception as e:
        print(f"   ❌ Error getting user info: {e}")
        return False

    # Check repository access
    print(f"\n2. Repository Access Test:")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"   ✓ Can read repository: {repo_id}")
        print(f"   - Private: {repo_info.private}")
        print(f"   - Last modified: {repo_info.last_modified}")
    except Exception as e:
        print(f"   ❌ Cannot access repository: {e}")
        return False

    # Test write permission by attempting to create a small test file
    print(f"\n3. Write Permission Test:")
    print(f"   Attempting to upload a test file...")

    try:
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test write permission - can be deleted\n")
            test_file = f.name

        # Try to upload
        api.upload_file(
            path_or_fileobj=test_file,
            path_in_repo="test_write_permission.txt",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Test write permission (can be deleted)"
        )

        print(f"   ✅ SUCCESS! You can write to {repo_id}")
        print(f"   Test file uploaded: test_write_permission.txt")
        print(f"   You can delete it at: https://huggingface.co/datasets/{repo_id}")

        # Clean up local test file
        import os
        os.unlink(test_file)

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED: Cannot write to repository")
        print(f"   Error: {error_msg}")

        if "403" in error_msg or "Forbidden" in error_msg:
            print("\n   Reason: Permission denied (403 Forbidden)")
            print("   - Your role in Salesforce org might still be 'read'")
            print("   - Or you need to be added as a collaborator to this specific repo")
        elif "404" in error_msg:
            print("\n   Reason: Repository not found or no access")

        return False

    print("\n" + "=" * 70)

if __name__ == "__main__":
    success = test_write_permission()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    if success:
        print("✅ You HAVE write access to Salesforce/ST-Evidence-Instruct")
        print("   You can now upload your dataset!")
    else:
        print("❌ You DO NOT have write access yet")
        print("\nNext steps:")
        print("1. Contact Salesforce HuggingFace organization admin")
        print("2. Request write access to Salesforce/ST-Evidence-Instruct")
        print("3. Or request your org role to be upgraded from 'read' to 'write'")

    print("=" * 70)

    sys.exit(0 if success else 1)
