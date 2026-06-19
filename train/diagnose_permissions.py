#!/usr/bin/env python3
"""
Diagnose HuggingFace permissions for Salesforce/ST-Evidence-Instruct
"""

from huggingface_hub import HfApi, whoami
import sys

def main():
    print("=" * 60)
    print("HuggingFace Permission Diagnostics")
    print("=" * 60)

    try:
        api = HfApi()

        # Check current user
        print("\n1. Current User Info:")
        user_info = whoami()
        print(f"   Username: {user_info['name']}")
        print(f"   Type: {user_info.get('type', 'N/A')}")
        print(f"   Organizations: {user_info.get('orgs', [])}")

        # Check token permissions
        print("\n2. Token Auth:")
        auth_info = user_info.get('auth', {})
        print(f"   Access token: {'***' + user_info.get('name', '')[-4:] if user_info.get('name') else 'N/A'}")
        print(f"   Type: {auth_info.get('type', 'unknown')}")

        # Try to access the repository
        print("\n3. Repository Access Test:")
        repo_id = "Salesforce/ST-Evidence-Instruct"

        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
            print(f"   ✓ Can read repository: {repo_id}")
            print(f"   - Private: {repo_info.private}")
            print(f"   - Author: {repo_info.author}")
            print(f"   - Last modified: {repo_info.last_modified}")
        except Exception as e:
            print(f"   ✗ Cannot access repository: {e}")
            return 1

        # Try to check write permissions (attempt to create a commit)
        print("\n4. Write Permission Test:")
        try:
            # This won't actually create a commit, just checks if we can
            print(f"   Testing write access to {repo_id}...")

            # Try to list repo files (this requires read access)
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            print(f"   ✓ Can list {len(files)} files in repository")

            # The actual write test would fail, but we can check org membership
            orgs = [org['name'] for org in user_info.get('orgs', [])]
            if 'Salesforce' in orgs:
                print(f"   ✓ You are a member of Salesforce organization")
                print(f"   ⚠️  But membership doesn't guarantee write access!")
                print(f"   ⚠️  Organization admins control repository permissions")
            else:
                print(f"   ✗ You are NOT a member of Salesforce organization")
                return 1

        except Exception as e:
            print(f"   ✗ Write test failed: {e}")
            return 1

        # Recommendations
        print("\n5. Recommendations:")
        print("   To fix 403 Forbidden errors, you need to:")
        print("   ")
        print("   Option A: Get repository write access")
        print("   - Contact Salesforce organization admins")
        print("   - Request write access to: Salesforce/ST-Evidence-Instruct")
        print("   - They need to add you as a collaborator with write permissions")
        print("   ")
        print("   Option B: Verify your token")
        print("   - Go to: https://huggingface.co/settings/tokens")
        print("   - Make sure your token has 'Write' type (not just 'Read')")
        print("   - Token should have scope: 'Write access to contents of repos'")
        print("   ")
        print("   Option C: Check organization settings")
        print("   - Ask admins if the organization requires special permissions")
        print("   - Some orgs restrict direct uploads and require PRs")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
